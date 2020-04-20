import torch
from options import TestOptions
from dataset import dataset_single
from model import DRIT
from saver import save_imgs
import os

def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()

  # data loader
  print('\n--- load dataset ---')
  datasetA = dataset_single(opts, 'A', opts.input_dim_a)
  datasetB = dataset_single(opts, 'B', opts.input_dim_b)
  if opts.a2b:
    loader = torch.utils.data.DataLoader(datasetA, batch_size=1, num_workers=opts.nThreads)
    loader_attr = torch.utils.data.DataLoader(datasetB, batch_size=1, num_workers=opts.nThreads, shuffle=True)
  else:
    loader = torch.utils.data.DataLoader(datasetB, batch_size=1, num_workers=opts.nThreads)
    loader_attr = torch.utils.data.DataLoader(datasetA, batch_size=1, num_workers=opts.nThreads, shuffle=True)

  # model
  print('\n--- load model ---')
  model = DRIT(opts)
  model.setgpu(opts.gpu)
  model.resume(opts.resume, train=False)
  model.eval()

  # directory
  result_dir = os.path.join(opts.result_dir, opts.name)
  if not os.path.exists(result_dir):
    os.mkdir(result_dir)

  # test
  print('\n--- testing ---')
  for idx1, (img1, img1_path) in enumerate(loader):
    print('{}/{}'.format(idx1, len(loader)))
    img1_path = img1_path[0]
    img1_prefix = os.path.basename(img1_path).split('.')[0]
#    print('img1_prefix:', img1_prefix)
    img1 = img1.cuda()
    imgs = [img1]
#    print('img1 type:', type(img1))
    names = [f'{img1_prefix}_input']
    for idx2, (img2, img2_path) in enumerate(loader_attr):
      img2_path = img2_path[0]
      img2_prefix = os.path.basename(img2_path).split('.')[0]
#      print('img2_prefix:', img2_prefix)
      if img1_prefix == img2_prefix:
        img2 = img2.cuda()
        imgs.append(img2)
        names.append(f'{img2_prefix}_real')
#        print('img2 type:', type(img2))
        with torch.no_grad():
          if opts.a2b:
            img = model.test_forward_transfer(img1, img2, a2b=True)
          else:
            img = model.test_forward_transfer(img2, img1, a2b=False)
        imgs.append(img)
        names.append(f'{img2_prefix}_fake')
        break
    save_imgs(imgs, names, result_dir)

  return

if __name__ == '__main__':
  main()
