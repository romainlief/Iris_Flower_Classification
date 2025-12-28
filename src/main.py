from parser import data_parser

if __name__=="__main__":
    parser = data_parser.DataParser("dataset/iris.csv")
    parser.download()
    print(parser.parse())