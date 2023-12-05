import time

class ExampleModel:
    epochs = 0
    data = [0,0,0]#body height, body weight, total samples
    last_change = time.time()
    
    @classmethod
    def get_model_parameters(cls):
        if cls.last_change + 10 < time.time():
            cls.last_change = time.time()
            cls.epochs += 1
        avg_weight = 0
        avg_height = 0
        if cls.data[2] > 0:
            avg_weight = cls.data[1]/cls.data[2]
            avg_height = cls.data[0]/cls.data[2]
        data = {"epochs": cls.epochs, "total_height": cls.data[0], 
                "total_weight": cls.data[1], "samples": cls.data[2],
                "average_weight": avg_weight, "average_height": avg_height}
        print(data)
        return data

    @classmethod
    def update_model(cls, data, epoch):
        if epoch == cls.epochs:
            cls.data[0] += data['height']
            cls.data[1] += data['weight']
            cls.data[2] += 1
            return True
        return False



