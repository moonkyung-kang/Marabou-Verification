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
    """Neural Network Verifier"""
    
    def __init__(self, network_path):
        """
        Args:
            network_path (str): file path (.onnx, .nnet, .pb 포맷 지원)
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
        """Robustness verification - 입력의 L∞ 섭동에 대한 출력 안정성 검증
        
        Args:
            input_point (np.array): input point
            epsilon (float): L∞ perturbation size
            target_label (int, optional): verification for a specific target label
        
        Returns:
            dict: 검증 결과 포함한 딕셔너리
        """
        # set constraint for input variable
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
        

    def verify_output_bounds(self, input_bounds):
        """Verification of output bounds for input range
        
        Args:
            input_bounds (list of tuple): (lower bound, upper bound) tuple list for each input variable
            
        Returns:
            dict: dictionary containing verification results
        """
        # set input range
        
        inputVars = self.network.inputVars[0][0]
        outputVars = self.network.outputVars[0][0]
        for h in range(inputVars.shape[0]):
            for w in range(inputVars.shape[1]):
                (lb, ub) = input_bounds[h][w]
                self.network.setLowerBound(inputVars[h][w],lb)
                self.network.setUpperBound(inputVars[h][w], ub)
        
        # calculate output bounds
        exitCode, bounds, stats = self.network.calculateBounds()
        print(exitCode)
        return {
            'result': exitCode,
            'output_bounds': bounds if exitCode == "" else None,
            'stats': stats
        }

    def verify_gelu_properties(self, input_range=(-5, 5), num_points=1000):
        """Verification of GELU activation function properties
        
        Args:
            input_range (tuple): input range
            num_points (int): number of points to check
        
        Returns:
            dict: verification results
        """
        # verify main properties of GELU
        properties = {
            'monotonic': True,  # monotonic increase
            'bounded_ratio': True,  # ratio between input and output is within [0, 1]
            'symmetric': True  # symmetric about the origin
        }
        
        # generate test points
        test_points = np.linspace(input_range[0], input_range[1], num_points)
        
        for x in test_points:
            # generate input variable
            input_var = self.network.getNewVariable()
            output_var = self.network.getNewVariable()
            
            # add GeLU constraint
            self.network.addGeLUConstraint(input_var, output_var)
        
            
            # 1. verify monotonicity
            if properties['monotonic']:
                # x1 < x2 이면 GeLU(x1) < GeLU(x2)
                x2 = x + 0.1
                input_var2 = self.network.getNewVariable()
                output_var2 = self.network.getNewVariable()
                self.network.addGeLUConstraint(input_var2, output_var2)
                
                # set constraint
                self.network.setLowerBound(input_var, x)
                self.network.setUpperBound(input_var, x)
                self.network.setLowerBound(input_var2, x2)
                self.network.setUpperBound(input_var2, x2)
                
                # output_var2 <= output_var 인 반례 찾기
                self.network.addInequality([output_var, output_var2], [1, -1], 0)
                
                exitCode, vals, _ = self.network.solve()
                if exitCode == "sat":
                    properties['monotonic'] = False
            
            # 2. verify output range
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
            
            # 3. verify symmetry
            if properties['symmetric']:
                neg_input_var = self.network.getNewVariable()
                neg_output_var = self.network.getNewVariable()
                
                self.network.addGeLUConstraint(neg_input_var, neg_output_var)
                # verify symmetry of output for -x and x
                self.network.setLowerBound(input_var, x)
                self.network.setUpperBound(input_var, x)
                self.network.setLowerBound(neg_input_var, -x)
                self.network.setUpperBound(neg_input_var, -x)
                
                # verify GeLU(-x) = -GeLU(x)
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
    # sys.stdout = open('fmnist_GELU_result_0.01.txt', 'w')
    # 사용 예시
    verifier = NetworkVerifier("./fmnist_GeLU_model.onnx")
    
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
        epsilon=0.1,
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
    
    
    # sys.stdout.close()
if __name__ == "__main__":
    main()
