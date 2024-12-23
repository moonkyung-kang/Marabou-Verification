import torch.onnx
import os
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
# 파이썬 환경 변수 설정
import sys
sys.path.insert(1, '/app/Marabou')
# sys.path.insert(1, '/Users/moonkyung/Desktop/marabou-docker/Marabou')
from maraboupy import Marabou
from maraboupy import MarabouCore
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
from maraboupy.MarabouUtils import Equation

# 옵션 설정, 모델 파일 경로 설정, 데이터 전처리
options = Marabou.createOptions(verbosity = 0)
model_filename = "./fmnist_dnn.onnx"
# model_filename= "./fmnist_dnn.onnx"
transform = transforms.Compose([
        transforms.ToTensor(),
    ])
training_data = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
test_data = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

train_dataloader=DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 모델 로드
network = Marabou.read_onnx(model_filename)
inputVars = network.inputVars[0][0]
outputVars = network.outputVars[0][0]

# 첫번째 이미지 로드
iterator = iter(train_dataloader)
inputs, classes = next(iterator)
image = inputs[0].numpy()[0]
correct_class = classes[0].item()

# 모델 출력
network_output = network.evaluateWithoutMarabou([image])[0]
predicted_class = np.argmax(network_output)

# 모델 구조 출력
print("inputVars structure:", inputVars)
print("outputVars structure:", outputVars)
print("network.outputVars structure:", network.outputVars)
print("inputVars structure shape:", inputVars.shape)
print("outputVars structure shape:", outputVars.shape)
print("inputVars structure type:", type(inputVars))
print("outputVars structure type:", type(outputVars))

# 첫번째 이미지의 실제 정답과 예측 값 출력
plt.title(f"correct:{correct_class}, predicted: {predicted_class}")
plt.imshow(image.reshape(28,28), cmap='gray')

# Epsilon을 이용해 이미지에 대한 bound 설정
epsilon = 0.1
for h in range(inputVars.shape[0]):
  for w in range(inputVars.shape[1]):
    network.setLowerBound(inputVars[h][w], image[h][w] - epsilon)
    network.setUpperBound(inputVars[h][w], image[h][w] + epsilon)

# 결과 초기화
result={}
# sys.stdout = open('fmnist_dnn_result_test.txt', 'w')
start_time = time.time()
for i in range(outputVars.shape[0]):
  # 정답이 아닌 클래스에 대해 max constraint 추가
  if i != correct_class:
    # add max constraint
    network.addMaxConstraint(set(outputVars), outputVars[i])
    # solve
    exit_code, vals, stats = network.solve(verbose = False, options = options)
    # if solution found, break
    print(exit_code)
    if len(vals) > 0:
      for j, var in enumerate(outputVars):
        print(f"output {j}: {vals[var]}")
      print(f"maxclass: {i}")
      result['result'] = exit_code
      result['counterexample'] = vals if exit_code == "sat" else None
      result['stats'] = stats
      break

# unsat 결과 출력
if len(result) == 0:
  result['result'] = 'unsat'
  result['counterexample'] = None
  result['stats'] = ""
end_time = time.time()
print("Robustness verification 결과:", result['result'])
print(f"Robustness verification 시간: {end_time - start_time:.4f}초")



result = {}
start_time = time.time()
input_bounds = [[(0, 1) for _ in range(28)] for _ in range(28)]  # (28,28) 형태의 2D 배열
inputVars = network.inputVars[0][0]
outputVars = network.outputVars[0][0]
for h in range(inputVars.shape[0]):
    for w in range(inputVars.shape[1]):
        (lb, ub) = input_bounds[h][w]
        network.setLowerBound(inputVars[h][w],lb)
        network.setUpperBound(inputVars[h][w], ub)
# calculate output bounds
exitCode, bounds, stats = network.calculateBounds()
print(exitCode)
result['result'] = exitCode
result['output_bounds'] = bounds if exitCode == "" else None
result['stats'] = stats
end_time = time.time()
print("Output bound verification 결과:", result['result'])
print(f"Output bound verification 시간: {end_time - start_time:.4f}초")

# ReLU의 속성 검증
start_time = time.time()
input_range=(-5, 5)
num_points=1000
properties = {
    'non_negative': True,  # output is always non-negative
    'identity_positive': True  # f(x) = x for x > 0
}
 # generate test points
test_points = np.linspace(input_range[0], input_range[1], num_points)

for x in test_points:
    # generate input variable
    input_var = network.getNewVariable()
    output_var = network.getNewVariable()
    
    # 1. verify non-negativity
    if properties['non_negative']:
        network.setLowerBound(input_var, x)
        network.setUpperBound(input_var, x)
        
        # ReLU 출력이 음수인 반례 찾기
        network.setUpperBound(output_var, 0)
        
        exitCode, _, _ = network.solve()
        if exitCode == "sat":
            properties['non_negative'] = False
    
    # 2. verify identity for positive inputs
    if properties['identity_positive'] and x > 0:
        network.setLowerBound(input_var, x)
        network.setUpperBound(input_var, x)
        
        # ReLU(x) = x for x > 0 검증
        e = Equation()
        e.addAddend(1, output_var)
        e.addAddend(-1, input_var)
        e.setScalar(0)
        network.addEquation(e)
        
        exitCode, _, _ = network.solve()
        if exitCode == "sat":
            properties['identity_positive'] = False
end_time = time.time()          
print("- non_negative:", properties['non_negative'])
print("- identity_positive:", properties['identity_positive'])
print(f"GELU Property verification 시간: {end_time - start_time:.4f}초")


# sys.stdout.close()


# def get_image_from_marabou(vals, inputVariables):
#   adversarial_image = [[] for _ in range(inputVariables.shape[1])]
#   for h in range(inputVariables.shape[0]):
#     for w in range(inputVariables.shape[1]):
#       adversarial_image[h].insert(w, vals[inputVariables[h][w]])
#   adversarial_image = np.array(adversarial_image)

#   return adversarial_image


# adversarial_image = get_image_from_marabou(vals, inputVars)
# network_output = network.evaluateWithoutMarabou([adversarial_image])[0]

# predicted_class = np.argmax(network_output)

# plt.title(f"correct:{correct_class}, predicted: {predicted_class}")
# plt.imshow(adversarial_image, cmap='gray')
# plt.show()