import csv

class user:
    def __init__(self):
        self.user_id = {}
        self.user_like = {}
        self.read_user()

    def read_user(self):
        row_id = 0
        with open('C:/Users/20213/PycharmProjects/graph_embedding/data/userNode.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.user_id[row_id] = row[0]
                self.user_like[row_id] = row[1:]
                row_id+=1

    def return_user(self):
        return self.user_id, self.user_like
