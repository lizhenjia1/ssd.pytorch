import sys
sys.path.append(".")
from data import *
from evaluation import eval_results
from data import CARPLATE_CLASSES as labelmap
from ssd import build_ssd
import torch
import torch.backends.cudnn as cudnn

# parameters
dataset = "CARPLATE"
dataset_root = "/data/TILT/1080p/carplate_only/"
cuda = True
model_path = "weights/carplate_weights/ssd512_20000_1080p.pth"
input_size = 512
top_k = 200
confidence_threshold = 0.01
eval_save_folder = "eval/"
obj_type = "carplate"

# load dataset for evaluation
eval_dataset = CARPLATEDetection(root=dataset_root, transform=BaseTransform(input_size, MEANS),
    target_transform=CARPLATEAnnotationTransform(keep_difficult=True), dataset_name='test')
# load net for evaluation
num_classes = len(labelmap) + 1  # +1 for background
eval_net = build_ssd('test', input_size, num_classes)  # initialize SSD
eval_net.load_state_dict(torch.load(model_path))
eval_net.eval()
print('Finished loading model!')
if cuda:
    eval_net = eval_net.cuda()
    cudnn.benchmark = True
# evaluation begin
eval_results.test_net(eval_save_folder, obj_type, dataset_root, 'test', labelmap, eval_net, cuda,
    eval_dataset, BaseTransform(eval_net.size, MEANS), top_k, input_size, thresh=confidence_threshold)