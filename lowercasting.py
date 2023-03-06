import os

def lowercasting(txtfile):
    with open(txtfile, 'r', encoding='utf-8') as file:
        data = file.read()

    data_lower = data.lower()

    # Print the lowercased text
    # print(data_lower)
    with open(txtfile, 'w', encoding='utf-8') as f:
        f.write(data_lower)
    
    # Path of the file
    print(f"Lowercased text written to file: {os.path.abspath(txtfile)}")

if __name__ == '__main__':
    txtfile = "./New-Dataset/small-117M-k40.train.txt"
    lowercasting(txtfile)