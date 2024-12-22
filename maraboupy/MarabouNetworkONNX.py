'''
Top contributors (to current version):
    - Kyle Julian
    - Haoze Wu
    - Teruhiro Tagomori
    - Tobey Shim
    - Idan Refaeli

This file is part of the Marabou project.
Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
in the top-level source directory) and their institutional affiliations.
All rights reserved. See the file COPYING in the top-level source
directory for licensing information.

MarabouNetworkONNX represents neural networks with piecewise linear constraints derived from the ONNX format
'''
import onnx
import onnxruntime
from maraboupy.MarabouNetwork import MarabouNetwork
from maraboupy.parsers.ONNXParser import ONNXParser
import os
from maraboupy import MarabouCore
from maraboupy.MarabouUtils import Equation

class MarabouNetworkONNX(MarabouNetwork):
    """Constructs a MarabouNetworkONNX object from an ONNX file

    Args:
        filename (str): Path to the ONNX file
        inputNames: (list of str, optional): List of node names corresponding to inputs
        outputNames: (list of str, optional): List of node names corresponding to outputs

    Returns:
        :class:`~maraboupy.Marabou.marabouNetworkONNX.marabouNetworkONNX`
    """
    def __init__(self, filename, inputNames=None, outputNames=None):
        super().__init__()
        self.readONNX(filename, inputNames, outputNames)

    def readONNX(self, filename, inputNames=None, outputNames=None, preserveExistingConstraints=False):
        if not preserveExistingConstraints:
            self.clear()

        self.filename = filename
        self.graph = onnx.load(filename).graph

        # Setup input node names
        if inputNames is not None:
            # Check that input are in the graph
            for name in inputNames:
                if not len([nde for nde in self.graph.node if name in nde.input]):
                    raise RuntimeError("Input %s not found in graph!" % name)
            self.inputNames = inputNames
        else:
            # Get default inputs if no names are provided
            assert len(self.graph.input) >= 1
            initNames = [node.name for node in self.graph.initializer]
            self.inputNames = [inp.name for inp in self.graph.input if inp.name not in initNames]

        # Setup output node names
        if outputNames is not None:
            if isinstance(outputNames, str):
                outputNames = [outputNames]

            # Check that outputs are in the graph
            for name in outputNames:
                if not len([nde for nde in self.graph.node if name in nde.output]):
                    raise RuntimeError("Output %s not found in graph!" % name)
            self.outputNames = outputNames
        else:
            # Get all outputs if no names are provided
            assert len(self.graph.output) >= 1
            initNames = [node.name for node in self.graph.initializer]
            self.outputNames = [out.name for out in self.graph.output if out.name not in initNames]

        ONNXParser.parse(self, self.graph, self.inputNames, self.outputNames)

    def getNode(self, nodeName):
        """Find the node in the graph corresponding to the given name

        Args:
            nodeName (str): Name of node to find in graph

        Returns:
            (str): ONNX node named nodeName

        :meta private:
        """
        nodes = [node for node in self.graph.node if nodeName in node.output]
        assert len(nodes) == 1
        return nodes[0]

    def splitNetworkAtNode(self, nodeName, networkNamePreSplit=None,
                           networkNamePostSplit=None):
        """
        Cut the current onnx file at the given node to create two networks.
        The output of the first network is the output of the given node.
        The second network expects its input to be the output of the first network.

        Return True if the split is successful.

        Args:
            nodeName (str): Name of node at which we want to cut the network
            networkNamePreSplit(str): If given, store the pre-split network at the given path.
            networkNamePostSplit(str): If given, store the post-split network at the given path.

        :meta private:
        """
        outputName = self.getNode(nodeName).output[0]
        if networkNamePreSplit is not None:
            try:
                onnx.utils.extract_model(self.filename, networkNamePreSplit,
                                         input_names=self.inputNames,
                                         output_names=[outputName])
            except Exception as error:
                print("Error when trying to create pre-split network: ", error)
                if os.path.isfile(networkNamePreSplit):
                    os.remove(networkNamePreSplit)
                return False
        if networkNamePostSplit is not None:
            try:
                onnx.utils.extract_model(self.filename, networkNamePostSplit,
                                         input_names=[outputName],
                                         output_names=self.outputNames)
            except Exception as error:
                print("Error when trying to create post-split network: ", error)
                if os.path.isfile(networkNamePostSplit):
                    os.remove(networkNamePostSplit)
                return False

        self.outputNames = [outputName]
        return True

    def evaluateWithoutMarabou(self, inputValues):
        """Try to evaluate the network with the given inputs using ONNX

        Args:
            inputValues (list of numpy array): Input values representing inputs to network

        Returns:
            (list of numpy array): Output values of neural network
        """
        # Check that all input variables are designated as inputs in the graph
        # Unlike Tensorflow, ONNX only allows assignment of values to input/output nodes
        onnxInputNames = [node.name for node in self.graph.input]
        for inName in self.inputNames:
            if inName not in onnxInputNames:
                raise NotImplementedError("ONNX does not allow intermediate layers to be set as inputs!")

        # Check that the output variable is designated as an output in the graph
        # Unlike Tensorflow, ONNX only allows assignment of values to input/output nodes
        onnxOutputNames = [node.name for node in self.graph.output]
        for outputName in self.outputNames:
            if outputName not in onnxOutputNames:
                raise NotImplementedError("ONNX does not allow intermediate layers to be set as the output!")

        initNames =  [node.name for node in self.graph.initializer]
        graphInputs = [inp.name for inp in self.graph.input if inp.name not in initNames]
        if len(inputValues) != len(graphInputs):
            raise RuntimeError("There are %d inputs to network, but only %d input arrays were given."%(len(graphInputs), len(inputValues)))

        # Use onnxruntime session to evaluate the point
        sess = onnxruntime.InferenceSession(self.filename)
        input_dict = dict()
        for i, inputName in enumerate(self.inputNames):

            # Try to cast input to correct type
            onnxType = sess.get_inputs()[i].type
            if 'float' in onnxType:
                inputType = 'float32'
            else:
                raise NotImplementedError("Inputs to network expected to be of type 'float', not %s" % onnxType)
            input_dict[inputName] = inputValues[i].reshape(self.inputVars[i].shape).astype(inputType)
        return sess.run(self.outputNames, input_dict)
    
    def addGeLUConstraint(self, input_var, output_var):
        """GeLU activation function에 대한 제약조건 추가
        
            GeLU(x) = 0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715x^3)))
            이를 구간별 Linear function으로 근사
            
            Args:
                input_var (int): 입력 변수 번호
                output_var (int): 출력 변수 번호
        """
        # 중간 변수들 생성
        temp_var1 = self.getNewVariable()  # x^3 항을 위한 변수
        temp_var2 = self.getNewVariable()  # tanh 입력을 위한 변수
        temp_var3 = self.getNewVariable()  # tanh 출력을 위한 변수
        
        # 상수들
        SQRT_2_PI = 0.7978845608028654  # sqrt(2/pi)
        COEF = 0.044715
        
        # 1. x^3 근사
        self.addPiecewiseLinearConstraint(input_var, temp_var1, [
            (-float('inf'), -2.0, -8.0, 0.0),
            (-2.0, -1.0, -1.0, -3.0),
            (-1.0, 1.0, 1.0, 0.0),
            (1.0, 2.0, 1.0, 3.0),
            (2.0, float('inf'), 8.0, 0.0)
        ])
        
        # 2. tanh 입력 계산: sqrt(2/pi) * (x + 0.044715x^3)
        e = Equation()
        e.addAddend(SQRT_2_PI, input_var)
        e.addAddend(SQRT_2_PI * COEF, temp_var1)
        e.addAddend(-1, temp_var2)
        e.setScalar(0)
        e.type = MarabouCore.Equation.EquationType.EQ
        self.addEquation(e)
        
        # 3. tanh 근사
        self.addPiecewiseLinearConstraint(temp_var2, temp_var3, [
            (-float('inf'), -2.0, 0.0, -1.0),
            (-2.0, -1.0, 0.4, -0.6),
            (-1.0, 1.0, 0.8, 0.0),
            (1.0, 2.0, 0.4, 0.6),
            (2.0, float('inf'), 0.0, 1.0)
        ])
        
        # 4. 최종 GeLU 출력 계산: 0.5x * (1 + tanh)
        e = Equation()
        e.addAddend(0.5, input_var)
        e.addAddend(0.5, input_var)
        e.addAddend(-1, output_var)
        e.addAddend(0.5, temp_var3)
        e.setScalar(0)
        e.type = MarabouCore.Equation.EquationType.EQ
        self.addEquation(e)

    def addPiecewiseLinearConstraint(self, input_var, output_var, points):
        """구간별 선형 제약조건 추가
        
        Args:
            input_var (int): 입력 변수 번호
            output_var (int): 출력 변수 번호 
            points (list): [(x1,x2,y1,y2), ...] 형태의 구간 정의 리스트
                         각 tuple은 (구간 시작, 구간 끝, 시작 y값, 끝 y값)을 의미
        """
        # MarabouCore.PiecewiseLinearConstraint 대신 기본 선형 제약조건들을 사용
        for x1, x2, y1, y2 in points:
            if x1 == float('-inf'):
                # 왼쪽 무한대 구간
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                e = Equation()
                e.addAddend(-slope, input_var)
                e.addAddend(1, output_var)
                e.setScalar(y2 - slope * x2)
                self.addEquation(e)
            elif x2 == float('inf'):
                # 오른쪽 무한대 구간  
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                e = Equation()
                e.addAddend(-slope, input_var)
                e.addAddend(1, output_var)
                e.setScalar(y1 - slope * x1)
                self.addEquation(e)
            else:
                # 나머지 구간
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                e = Equation()
                e.addAddend(-slope, input_var)
                e.addAddend(1, output_var)
                e.setScalar(y1 - slope * x1)
                self.addEquation(e)