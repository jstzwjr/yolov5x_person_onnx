import onnxruntime as ort
from functools import partial
from myutils import non_max_suppression, scale_coords, letterbox
import torch
import os
import numpy as np
import cv2
'''
OnnxInferenceSession使用方法
实例化时需要传入onnx模型路径和预处理函数，预处理函数返回值需要为字典格式，
形如：{"input1":np.ndarray(b,c,h,w),"input2":np.ndarray(b,c,h,w),...}
模型的推理通过调用__call__函数实现，返回一个列表

'''
img_size = 640
num_classes = 80
person_det_conf_thres = float(os.environ.get('person_iou_thres', 0.5))
person_det_iou_thres = float(os.environ.get('person_det_iou_thres', 0.6))
person_det_thres = float(os.environ.get('person_det_thres', 0.5))


def make_grid(nx, ny):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def init_grid():
    grid = []
    for ny, nx in [(img_size//32, img_size//32), (img_size//16, img_size//16), (img_size//8, img_size//8)]:
        grid.append(make_grid(nx, ny).cuda())
    return grid


grid = init_grid()
stride = torch.tensor([32., 16., 8.]).cuda()
anchor_grid = torch.tensor([[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [
                           10, 13, 16, 30, 33, 23]]).view(3, 1, -1, 1, 1, 2).contiguous().cuda()
names = ["person"]


class OnnxInferenceSession:
    def __init__(self, model_path, preprocess_func=lambda x: {"images": x},):
        self.sess = ort.InferenceSession(model_path)
        # self.preprocess = preprocess_func

    def preprocess(self, im0s, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        img = letterbox(im0s, new_shape, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to("cuda:0")
        # img = img.half()
        img = img / 255.0
        img = img.unsqueeze(0)
        img = img.cpu().numpy()
        return {"images": img}

    def __call__(self, inputs):
        # return list of results
        model_inputs = self.preprocess(inputs)
        onnx_outputs = self.sess.run(None, model_inputs)
        result = self.post_process(inputs, onnx_outputs)
        return result

    def post_process(self, im0s, x):
        '''
        x为输出fm，大小分别为bx(3n_c)x20x20,bx(3n_c)x40x40,bx(3n_c)x80x80
        '''
        for i in range(len(x)):
            x[i] = torch.from_numpy(x[i])
            x[i] = x[i].view((-1, 3, img_size//(stride[i].long()), img_size//(stride[i].long()),
                              num_classes+5)).contiguous().cuda()
        z = []
        for i in range(len(x)):
            # x(bs,255,20,20) to x(bs,3,20,20,85)
            bs, _, ny, nx, _ = x[i].shape
            # x[i] = x[i].view(bs, 3, 7, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if grid[i].shape[2:4] != x[i].shape[2:4]:
                grid[i] = make_grid(nx, ny).to(x[i].device)
            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 +
                           grid[i].to(x[i].device)) * stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
            # print(anchor_grid[i])
            z.append(y.view(bs, -1, num_classes+5))
        pred = torch.cat(z, 1)

        pred = non_max_suppression(pred, conf_thres=person_det_conf_thres,
                                   iou_thres=person_det_iou_thres, classes=None, agnostic=False)

        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    (img_size, img_size), det[:, :4], im0s.shape).round()
        res = pred[0]
        if res is None:
            res = []
        else:
            ind = (res[:, -2] >= person_det_thres) & (res[..., -1] == 0)
            res = res[ind]
            res = res.cpu().numpy()[:, :-1].tolist()
        return res


model = OnnxInferenceSession(os.path.join(
    os.path.dirname(__file__), "yolov5x_sim.onnx"))


def inference(img):
    outputs = model(img)
    return outputs


if __name__ == "__main__":
    from time import time
    img = cv2.imread('images/bus.jpg')

    for _ in range(100):
        s = time()
        outputs = model(img)
        e = time()
        print(e-s)
    print(outputs)
