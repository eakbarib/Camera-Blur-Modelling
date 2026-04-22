import shutil
from pathlib import Path
import subprocess
import sqlite3
import numpy as np
from PIL import Image
import pycolmap

# wrapper class for various colmap utilities
class colmapDB:
    def colmap_exec(self, args, verbose=False):
        arglist = args.split()
        arglist.insert(0, self.executable_path)
        try:
            subprocess.run(arglist, check=True, capture_output=not verbose)
            return True
        except subprocess.CalledProcessError as e:
            return False
    
    def __init__(self, database_path, colmap_executable=None, del_existing=False):
        self.database_path = database_path
        self.image_path = None
        self.output_path = None
        self.del_existing = del_existing
        
        # find execuatble
        if colmap_executable is not None:
            if not colmap_executable.exists():
                raise FileNotFoundError("Cannot find colmap executable at specified location.")
            self.executable_path = colmap_executable
        else:
            exe_path = shutil.which("colmap.exe")
            if not exe_path:
                raise FileNotFoundError("Cannot find colmap executable. Add colmap to Path or pass its path as an argument.")
            self.executable_path = Path(exe_path)
        
        # init database
        if del_existing:
            database_path.unlink(missing_ok=True)
            
        if not self.colmap_exec(f"database_creator --database_path {self.database_path}"):
            raise ChildProcessError("Failed to initialize database.")
        pass
        
    def add_images_auto(self, image_path):
        """
        Adds images to the database (actually just puts it off to reconstruction)
        Use for the vanilla colmap behavior
        """
        self.image_path = image_path
    
    def add_images_cameras(self, image_path, calibration_assigner):
        """
        Adds images and corresponding cameras to the database
        """
        self.image_path = image_path
        
        with sqlite3.connect(self.database_path) as db:
            cur = db.cursor()
            
            # check if images have already been added
            cur.execute("SELECT EXISTS (SELECT 1 FROM images);")
            if cur.fetchone()[0]:
                return
            
            for i, image in enumerate(image_path.iterdir()):
                identifier = i+1
                
                # add image
                cur.execute(
                    "INSERT INTO images (image_id, name, camera_id) VALUES (?, ?, ?)",
                    (identifier, image.name, identifier)
                )
                
                # add camera
                with Image.open(image) as img:
                    width, height = img.size
                
                K, D = calibration_assigner(image)
                params = np.array([K[0,0], K[1,1], K[0,2], K[1,2], D[0], D[1], D[2], D[3]], dtype=np.float64).tobytes()
                cam_type = pycolmap.CameraModelId.OPENCV.value
                
                cur.execute(
                    "INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length)"
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (identifier, cam_type, width, height, params, 1)
                )
                
                # add rig
                cur.execute(
                    "INSERT INTO rigs (rig_id, ref_sensor_id, ref_sensor_type) VALUES (?, ?, ?)",
                    (identifier, identifier, 0)
                )
                
            db.commit()
        
    def register_images(self, verbose=False):
        if self.image_path is None:
            raise Exception("No images added")
    
        if not self.colmap_exec(f"feature_extractor --database_path {self.database_path} --image_path {self.image_path}", verbose=verbose):
            raise ChildProcessError("Failed to extract features.")
        
    def match_images(self, verbose=False):
        if not self.colmap_exec(f"exhaustive_matcher --database_path {self.database_path}", verbose=verbose):
            raise ChildProcessError("Failed to extract features.")
        
    def reconstruct_sparse(self, output_path, verbose=False, refine=False):
        if output_path.exists():
            if self.del_existing:
                shutil.rmtree(output_path)
            else:
                self.output_path = output_path
                return
        
        if self.image_path is None:
            raise Exception("No images added")
        
        self.output_path = output_path
        output_path.mkdir(exist_ok=True)
        
        self.colmap_exec(f"mapper --database_path {self.database_path} --image_path {self.image_path} --output_path {output_path}{
                " --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0" if not refine else ""
            }", 
            verbose=verbose
        )
        
    def sparse_reproj_error(self):
        if self.output_path is None:
            raise Exception("Reconstruction not created yet. Run reconstruct() first")
        
        reproj_errors = []
        
        for reconstruction_path in self.output_path.iterdir():
            reconstruction = pycolmap.Reconstruction(reconstruction_path)
            reproj_errors.append(reconstruction.compute_mean_reprojection_error())
        
        return np.mean(reproj_errors)
        
    def reconstruct_dense(self, dense_output_path, verbose=False):        
        if dense_output_path.exists():
            if self.del_existing:
                shutil.rmtree(dense_output_path)
            else:
                self.dense_output_path = dense_output_path
                return
            
        if self.image_path is None:
            raise Exception("No images added")
        
        if self.output_path is None:
            raise Exception("No sparse reconstruction created")
        
        self.dense_output_path = dense_output_path
        dense_output_path.mkdir(exist_ok=True)
        
        self.colmap_exec(
            f"image_undistorter --image_path {self.image_path} --input_path {self.output_path / '0'} --output_path {self.dense_output_path}", 
            verbose=verbose
        )
        
        self.colmap_exec(
            f"patch_match_stereo --workspace_path {self.dense_output_path}"
            " --PatchMatchStereo.max_image_size 1080", 
            verbose=verbose
        )
        
        self.colmap_exec(
            f"stereo_fusion --workspace_path {self.dense_output_path} --output_path {self.dense_output_path / "fused.ply"}", 
            verbose=verbose
        )
        
        self.colmap_exec(
            f"poisson_mesher --input_path {self.dense_output_path / "fused.ply"} --output_path {self.dense_output_path / "meshed.ply"}", 
            verbose=verbose
        )
        