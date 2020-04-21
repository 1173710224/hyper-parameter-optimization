'''
ijcai group
author:
chargehand:陈泊舟
important members:张恺欣，欧龙燊，霸臣民
'''
import sys
sys.path.append('..')
'''
train method
input:a raw dataset,dataframe(it needs encoded)
output:none
describe:
encode the input dataset to get a feature vector V,
call train.core.predict(V) to get P(predicted best parameters),
optimize parameters from P with the local optimum algorithm,
save the model to "model" folder
'''
def train(dataset):
    import preprocess
    vector = preprocess.encode(dataset)
    import train.core
    predictParams = train.core.predict(vector)
    import opt
    baseParams = opt.run(predictParams,dataset)
    # 使用xgboost训练模型并保存

    return
'''
predict method
input: data is a single data,model is a file path directing to the model you would like to use
output:the predict result
describe:none
'''
def predict(data,model = 'model\default.model'):
    return
