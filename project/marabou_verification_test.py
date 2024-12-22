import sys
sys.path.insert(1, '/Users/moonkyung/Desktop/marabou-docker/Marabou')

from maraboupy import Marabou
from maraboupy.MarabouUtils import Equation
from maraboupy.MarabouNetwork import MarabouNetwork
from maraboupy.MarabouCore import addReluConstraint
import numpy as np
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
class NetworkVerifier:
    """신경망 검증을 위한 클래스"""
    
    def __init__(self, network_path):
        """
        Args:
            network_path (str): 신경망 파일 경로 (.onnx, .nnet, .pb 포맷 지원)
        """
        self.network_path = network_path
        self.network = self._load_network()
        
    def _load_network(self):
        """신경망 파일 로드"""
        ext = os.path.splitext(self.network_path)[1]
        
        if ext == '.onnx':
            return Marabou.read_onnx(self.network_path)
        elif ext == '.nnet':
            return Marabou.read_nnet(self.network_path)
        elif ext == '.pb':
            return Marabou.read_tf(self.network_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")

    def evaluateWithoutMarabou(self, inputValues):
        """네트워크 평가를 위한 메소드"""
        return self.network.evaluateWithoutMarabou(inputValues)

    def verify_robustness(self, input_point, epsilon, target_label=None):
        """강건성 검증 - 입력의 L∞ 섭동에 대한 출력 안정성 검증
        
        Args:
            input_point (np.array): 검증할 입력점
            epsilon (float): L∞ 섭동 크기
            target_label (int, optional): 특정 타겟 레이블에 대한 검증
        
        Returns:
            dict: 검증 결과 포함한 딕셔너리
        """
        # 입력 변수에 대한 제약조건 설정
        options = Marabou.createOptions(verbosity = 0)
        inputVars = self.network.inputVars[0][0]
        outputVars = self.network.outputVars[0][0]
        for h in range(inputVars.shape[0]):
            for w in range(inputVars.shape[1]):
                self.network.setLowerBound(inputVars[h][w], input_point[h][w] - epsilon)
                self.network.setUpperBound(inputVars[h][w], input_point[h][w] + epsilon)
        for i in range(outputVars.shape[0]):
            if i != target_label:
                # add max constraint
                self.network.addMaxConstraint(set(outputVars), outputVars[i])
                # solve
                exit_code, vals, stats = self.network.solve(verbose = False, options = options)
                # if solution found, break
                print(exit_code)
                if len(vals) > 0:
                    for j, var in enumerate(outputVars):
                        print(f"output {j}: {vals[var]}")
                    print(f"maxclass: {i}")
                    return {
                    'result': exit_code,
                    'counterexample': vals if exit_code == "sat" else None,
                    'stats': stats
                    }
        
        return {
            'result': 'unsat',
            'counterexample': None,
            'stats': ""
            }
        
        #             break 
        #         else:
        #             print("No counterexample found")
        #             print(stats)
        # # return {
        #     'result': exitCode,
        #     'counterexample': vals if exitCode == "sat" else None,
        #     'stats': stats
        # }
        # input_vars = self.network.inputVars[0].flatten()
        # for i, var in enumerate(input_vars):
        #     self.network.setLowerBound(var, max(0, input_point[i] - epsilon))
        #     self.network.setUpperBound(var, min(1, input_point[i] + epsilon))
        
        # # 출력 제약조건 설정 (타겟 레이블이 주어진 경우)
        # if target_label is not None:
        #     output_vars = self.network.outputVars[0].flatten()
        #     for i in range(len(output_vars)):
        #         if i != target_label:
        #             # 타겟 레이블의 출력이 다른 모든 출력보다 커야 함
        #             self.network.addInequality([output_vars[i], output_vars[target_label]], 
        #                                      [1, -1], 0)
        
        # # 검증 실행
        # exitCode, vals, stats = self.network.solve()
        
        # return {
        #     'result': exitCode,
        #     'counterexample': vals if exitCode == "sat" else None,
        #     'stats': stats
        # }

    def verify_property(self, property_file):
        """속성 파일을 이용한 검증
        
        Args:
            property_file (str): 속성 정의 파일 경로
            
        Returns:
            dict: 검증 결과 포함한 딕셔너리
        """
        exitCode, vals, stats = self.network.solve(propertyFilename=property_file)
        
        return {
            'result': exitCode,
            'counterexample': vals if exitCode == "sat" else None,
            'stats': stats
        }

    def verify_output_bounds(self, input_bounds):
        """입력 범위에 대한 출력 범위 검증
        
        Args:
            input_bounds (list of tuple): 각 입력 변수의 (하한, 상한) 튜플 리스트
            
        Returns:
            dict: 검증 결과 포함한 딕셔너리
        """
        # 입력 범위 설정
        
        inputVars = self.network.inputVars[0][0]
        outputVars = self.network.outputVars[0][0]
        for h in range(inputVars.shape[0]):
            for w in range(inputVars.shape[1]):
                (lb, ub) = input_bounds[h][w]
                self.network.setLowerBound(inputVars[h][w],lb)
                self.network.setUpperBound(inputVars[h][w], ub)
        
        # input_vars = self.network.inputVars[0].flatten()
        # for var, (lb, ub) in zip(input_vars, input_bounds):
        #     self.network.setLowerBound(var, lb)
        #     self.network.setUpperBound(var, ub)
        
        # 출력 범위 계산
        exitCode, bounds, stats = self.network.calculateBounds()
        print(exitCode)
        return {
            'result': exitCode,
            'output_bounds': bounds if exitCode == "" else None,
            'stats': stats
        }

    def verify_gelu_properties(self, input_range=(-5, 5), num_points=1000):
        """GeLU 활성화 함수의 속성 검증
        
        Args:
            input_range (tuple): 검증할 입력 범위
            num_points (int): 검사할 포인트 수
        
        Returns:
            dict: 검증 결과
        """
        # GeLU의 주요 속성들 검증
        properties = {
            'monotonic': True,  # 단조 증가
            'bounded_ratio': True,  # 입력과 출력의 비율이 [0, 1] 범위 내
            'symmetric': True  # 원점 대칭
        }
        
        # 테스트 포인트 생성
        test_points = np.linspace(input_range[0], input_range[1], num_points)
        
        for x in test_points:
            # 입력 변수 생성
            input_var = self.network.getNewVariable()
            output_var = self.network.getNewVariable()
            
            # GeLU 제약조건 추가
            self.network.addGeLUConstraint(input_var, output_var)
            
            # Relu 제약조건 추가
            # addReluConstraint(self.network, input_var, output_var)
            
            # 1. 단조성 검증
            if properties['monotonic']:
                # x1 < x2 이면 GeLU(x1) < GeLU(x2)
                x2 = x + 0.1
                input_var2 = self.network.getNewVariable()
                output_var2 = self.network.getNewVariable()
                self.network.addGeLUConstraint(input_var2, output_var2)
                # Relu 제약조건 추가
                # addReluConstraint(self.network, input_var2, output_var2)
                
                # 제약조건 설정
                self.network.setLowerBound(input_var, x)
                self.network.setUpperBound(input_var, x)
                self.network.setLowerBound(input_var2, x2)
                self.network.setUpperBound(input_var2, x2)
                
                # output_var2 <= output_var 인 반례 찾기
                self.network.addInequality([output_var, output_var2], [1, -1], 0)
                
                exitCode, vals, _ = self.network.solve()
                if exitCode == "sat":
                    properties['monotonic'] = False
            
            # 2. 출력 범위 검증
            if properties['bounded_ratio']:
                self.network.setLowerBound(input_var, x)
                self.network.setUpperBound(input_var, x)
                
                # 0 <= GeLU(x) <= x for x >= 0
                # x <= GeLU(x) <= 0 for x < 0
                if x >= 0:
                    self.network.setLowerBound(output_var, 0)
                    self.network.setUpperBound(output_var, x)
                else:
                    self.network.setLowerBound(output_var, x)
                    self.network.setUpperBound(output_var, 0)
                
                exitCode, _, _ = self.network.solve()
                if exitCode == "unsat":
                    properties['bounded_ratio'] = False
            
            # 3. 대칭성 검증
            if properties['symmetric']:
                neg_input_var = self.network.getNewVariable()
                neg_output_var = self.network.getNewVariable()
                
                self.network.addGeLUConstraint(neg_input_var, neg_output_var)
                # Relu 제약조건 추가
                # addReluConstraint(self.network, neg_input_var, neg_output_var)
                # -x와 x에 대한 출력이 대칭인지 검증
                self.network.setLowerBound(input_var, x)
                self.network.setUpperBound(input_var, x)
                self.network.setLowerBound(neg_input_var, -x)
                self.network.setUpperBound(neg_input_var, -x)
                
                # GeLU(-x) = -GeLU(x) 검증
                e = Equation()
                e.addAddend(1, output_var)
                e.addAddend(1, neg_output_var)
                e.setScalar(0)
                self.network.addEquation(e)
                
                exitCode, _, _ = self.network.solve()
                if exitCode == "sat":
                    properties['symmetric'] = False
        
        return properties

def main():
    sys.stdout = open('fmnist_GELU_result_0.01.txt', 'w')
    # 사용 예시
    verifier = NetworkVerifier("/Users/moonkyung/Desktop/Marabou/project/fmnist_GeLU_model.onnx")
    # verifier = NetworkVerifier("/Users/moonkyung/Desktop/Marabou/project/fmnist_dnn.onnx")
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
    # network = Marabou.read_onnx(model_filename)
    # inputVars = verifier.inputVars[0][0]
    # outputVars = verifier.outputVars[0][0]
    iterator = iter(train_dataloader)
    inputs, classes = next(iterator)
    image = inputs[0].numpy()[0]
    correct_class = classes[0].item()
    
    network_output = verifier.evaluateWithoutMarabou([image])[0]
    predicted_class = np.argmax(network_output)
    # 1. Robustness verification
    input_point = image  # (28,28)
    start_time = time.time()
    robustness_result = verifier.verify_robustness(
        input_point=input_point,
        epsilon=0.01,
        target_label=correct_class
    )
    end_time = time.time()
    print("Robustness verification 결과:", robustness_result['result'])
    print(f"Robustness verification 시간: {end_time - start_time:.4f}초")
    
    # 2. Output bound verification
    input_bounds = [[(0, 1) for _ in range(28)] for _ in range(28)]  # (28,28) 형태의 2D 배열
    start_time = time.time()
    bounds_result = verifier.verify_output_bounds(input_bounds)
    end_time = time.time()
    print("Output bound verification 결과:", bounds_result['result'])
    print(f"Output bound verification 시간: {end_time - start_time:.4f}초")
    
    # 3. GELU Property verification
    # verifier = NetworkVerifier("/Users/moonkyung/Desktop/Marabou/project/fmnist_GeLU_model.onnx")
    start_time = time.time()
    gelu_properties = verifier.verify_gelu_properties()
    end_time = time.time()
    print("GELU Property verification 결과:")
    print("- Monotonic:", gelu_properties['monotonic'])
    print("- Bounded ratio:", gelu_properties['bounded_ratio'])
    print("- Symmetry:", gelu_properties['symmetric'])
    print(f"GELU Property verification 시간: {end_time - start_time:.4f}초")
    
    
    sys.stdout.close()
if __name__ == "__main__":
    main()
