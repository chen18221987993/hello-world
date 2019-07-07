import numpy
 
def loadDataSet(file_Name):
    '''
    这个函数用来加载训练数据集
    输入：存储数据的文件名
    输出：数据集列表 以及 类别标签列表  
    '''
    with open (file_Name,encoding='utf8')as fr:
        dataMat=[];labelMat=[]
        line_list=fr.readlines()[1:]
        Feature_number=len(line_list[0].strip().split())
        for line in line_list:
            lineArr=[]
            line_new=line.strip().split()
            for i in range(Feature_number):
                lineArr.append(float(line_new[i]))
            dataMat.append(lineArr[1:-1])
            labelMat.append(lineArr[-1])
        return numpy.array(dataMat),numpy.array(labelMat)
    
def dataSplit(dataArr,labelArr):
    '''
    此函数的作用是将有限的样本数据集划分成训练集和测试集两部分
    输入：dataArr 所有的样本数据集，以numpy.ndarray形式封装
        labelArr 所有样本数据集对应的类别标签
    输出：trainDateArr 训练集
          trainlabelArr 训练集对应的类别标签
          testDateArr 测试集
          testlabelArr 测试集对应的类别标签
    '''
    trainDateArr = numpy.delete(dataArr.copy(),[1,9,12],0)
    trainlabelArr = numpy.delete(labelArr.copy(),[1,9,12])
    testDateArr = dataArr.copy()[[1,9,12],:]
    testlabelArr =labelArr.copy()[[1,9,12]]
    return  trainDateArr,trainlabelArr,testDateArr,testlabelAr

def treeClassify(dataMat, column, threshold, inequation):
    '''
    这是一个通过阈值threshold来对样本数据进行分类的，所有在阈值一边的数据会分到类别-1，而在另外一边的数据分到类别+1
    输入： datamat是样本数据集，接收列表形式的输入
           column用于指定待切分的特征
           threshold用来作为column所指定的特征列当中的值的比较阈值
           inequation用来指定是大于还是小于阈值
    输出：决策之后的类别标签（numpy.ndarray形式）
    '''
    if dataMat.ndim> 1:
        label_result = numpy.ones((dataMat.shape[0], 1))
    else:
        label_result = numpy.ones((1, 1))
    if inequation == 'less_than':
        if dataMat.ndim > 1:        #表示dataMat是一个二维数组
            label_result[numpy.nonzero(dataMat[:, column] <= threshold)[0]] = -1.0
        else:                       #否则dataMat就是一个一维向量
            if dataMat[column] <= threshold:
                label_result = -1.0
    else:
        if dataMat.ndim  > 1:       #表示dataMat是一个二维数组
            label_result[numpy.nonzero(dataMat[:, column] > threshold)[0]] = -1.0
        else:                        #否则dataMat就是一个一维向量
            if dataMat[column] > threshold:
                label_result = -1.0
    return label_result

def buildSingleTree(dataArr, labelArr, weightArr):
    '''
     这个函数根据加权错误率来找到最佳的单层决策树
     输入：dataArr是numpy.ndarray形式的样本数据集
           labelArr是numpy.ndarray形式的类别标签
           weightArr是当前轮次迭代所对应的权重向量
    输出：bestSigleTree 存储给定权重向量weightArr时所得到的最佳单层决策树的相关信息
          minError 最佳单层决策树所对应的错误率，即最小错误率
          bestLabelResult 当前最佳单层决策树的类别估计值
    '''
    dataMatrix = numpy.mat(dataArr)
    labelMatrix = numpy.mat(labelArr).T
    m, n = dataMatrix.shape
    bestSigleTree = {}
    bestLabelResult = numpy.mat(numpy.zeros((m, 1)))
    minError = numpy.inf
    for i in range(n):
        feature_value_list = sorted(dataMatrix.A[:, i])
        for j in feature_value_list:
            for inequation in ['less_than', 'greater_than']:
                threshold = float(j)
                predicted_label = treeClassify(dataMatrix, i, threshold, inequation)
                current_error = numpy.mat(numpy.ones((m, 1)))
                current_error[predicted_label == labelMatrix] = 0
                weighted_Error = weightArr.T * current_error
                if weighted_Error < minError:
                    minError = weighted_Error
                    bestLabelResult = predicted_label[:, :]
                    bestSigleTree['column'] = i
                    bestSigleTree['threshold'] = threshold
                    bestSigleTree['inequation'] = inequation
    return bestSigleTree, minError, bestLabelResult

def refreshWeightArr(expon, original_weightArr):
    '''
    这是一个训练数据集的权值分布WeightArr的更新函数
    输入：expon是由基分类器的权重（由分类错误率计算得到的）、类别标签以及单层决策树划分后得到的类别估计值计算得到的
         original_weightArr是更新之前的WeightArr
    输出：updated_weightArr是更新之后的训练数据集的权值分布
    '''
    intermediate_variable = original_weightArr * numpy.exp(expon)
    updated_weightArr = intermediate_variable / intermediate_variable.sum()
    return updated_weightArr

def adaBoostTrain(dataArr, labelArr, iterationsNumber=40):
    '''
    这是基于单层决策树的adaBoost算法
    输入：dataArr是numpy.ndarray形式的样本数据集
          labelArr是numpy.ndarray形式的类别标签
          iterationsNumber是迭代次数
    输出：多个弱分类器组成的numpy.ndarray形式的分类器集合
    '''
    classifier_Arr = []
    m = dataArr.shape[0]
    weightArr = numpy.ones((m, 1)) / m
    accumulative_estimated_value = numpy.zeros((m, 1))
    for i in range(iterationsNumber):
        current_SigleTree, current_Error, current_label_Result =\
                           buildSingleTree(dataArr, labelArr, weightArr)
        alpha = float(0.5 * numpy.log((1.0 - current_Error) / max(current_Error, 1e-16)))
        current_SigleTree['alpha'] = alpha
        classifier_Arr.append(current_SigleTree)
        expon = -1 * alpha * numpy.array(labelArr).reshape(m, 1) * current_label_Result
        weightArr = refreshWeightArr(expon, weightArr)
        accumulative_estimated_value += alpha * current_label_Result
        if len(numpy.nonzero(numpy.sign(accumulative_estimated_value) !=\
                             labelArr.reshape(m, 1))[0]) == 0:
            break
 
    return classifier_Arr


def testDateClassify(newDateSet, classifierArr):
    '''
    此函数用于对新数据进行自动分类,得到分类结果
    输入：newDateSet是待分类的新数据
        classifierArr是根据训练数据集用adaBoost算法训练出来的弱分类器集合
    输出：类别预测值
    '''
    newdataArr = numpy.array(newDateSet)
    if newdataArr.ndim > 1:
        m = newdataArr.shape[0]
    else:
        m = 1
    accumulative_estimated_value = numpy.zeros((m,1))
    for i in range(len(classifierArr)):
        current_label_Result = treeClassify(newdataArr, classifierArr[i]['column'], \
                        classifierArr[i]['threshold'], classifierArr[i]['inequation'])
        accumulative_estimated_value +=classifierArr[i]['alpha'] * current_label_Result
 
    return numpy.sign(accumulative_estimated_value)


if __name__=="__main__":
    dataArr,labelArr = loadDataSet('西瓜数据集.txt')
    trainDateArr, trainlabelArr, testDateArr, testlabelArr = dataSplit(dataArr, labelArr)
    classifierArr = adaBoostTrain(trainDateArr, trainlabelArr)
    test_result = testDateClassify(testDateArr, classifierArr)
    print('预测结果')
    print(test_result)                 #adaBoost算法对测试集类别的预测结果
    print('真实结果')
    print(testlabelArr.reshape(3,1))

    

