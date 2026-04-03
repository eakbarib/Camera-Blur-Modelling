# load parameters
# warp calib image + generate depth map
#   w2c*p2w
#   T = cam_mat*pose
# get blur radi from depth map
#   may require some kind of fitting
#   use a calib image with hoops (use manual blur radius estimation for now)
# apply blur (spatially variant)
# apply distortion

# formalize transforms & send to mahdi

# assumptions / models
# aperture radius is constant
# distortion is constant
# focal length is polynomial in inverse focus distance
# blur radius may be independent of focal length
# blur radius is linear with depth

# for some point on the image, (u,v,0,1), u in (-w/2,w/2), v in (-h/2,h/2)
# let Q be the 3x4 transform matrix (rotation + translation) from (u,v,0,1) to (x,y,z)
# let P be the 3x3 projection matrix from (x,y,z) to (du',dv',d)
# find the 3x3 homography matrix, H, from (u,v,1) to (du',dv',d)

# delete z column from Q, profit
# need to find differentible imwarp
# find derivative of image wrt rotation vector, translation vector, fx, fy, cx, cy
# update pose estimation to save rotation vector