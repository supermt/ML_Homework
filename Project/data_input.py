import numpy as np
def getdata(filename, train_percentage = 0.9):
  data_mat = []

  with open(filename, 'r') as f:
    data = f.readlines()
    for line in data:
      source = line.split(',')
      numbers_float = map(float, source)
      data_mat.append(numbers_float)

  data_mat = np.matrix(data_mat)

  features = (data_mat.T[:13]).T
  predicts = data_mat[:,13]

  total_size = data_mat.shape[0]
  train_size = int(train_percentage * total_size)
  predict_size = int(total_size - train_size)

  train_set_x = features[predict_size:]
  train_set_y = predicts[predict_size:]

  predict_set_x = features[:predict_size]
  target_y = predicts[:predict_size]

  return train_set_x,train_set_y,predict_set_x,target_y