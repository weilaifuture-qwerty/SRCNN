import os, random
import shutil

test = random.sample(os.listdir("/Users/weilai/Desktop/UIUC/FA24/CS444/project/coco/test2014"), k = 50)
for path in test:
    shutil.copyfile("/Users/weilai/Desktop/UIUC/FA24/CS444/project/coco/test2014/" + path, "./test/" + path)
