from imgSet import img_sets
import numpy as np
import cv2 as cv


# pre-generate focus stacks (significantly speeds up focus stack handling)

def gen_stack(set_id):
    img_set = img_sets[set_id]
    first, _ = img_set.read_img(0)
    stack = np.empty((img_set.count, first.shape[0], first.shape[1]), dtype=first.dtype)
    for i in range(img_set.count):
        img, _ = img_set.read_img(i)
        stack[i] = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    np.save(f"./stacks/{img_set.id}.npy", stack)
    print(f"done stack for {img_set.id}")

def gen_stacks():
    #gen_stack("or6_ir0_ds20"),
    gen_stack("or6_ir3_ds20"),
    gen_stack("or10_ir0_ds30"),
    gen_stack("or10_ir1_ds20"),
    gen_stack("or10_ir2_ds20"),
    gen_stack("or10_ir5_ds30"),
    gen_stack("or15_ir0_ds40"),
    gen_stack("or15_ir7_ds40")
    
gen_stacks()