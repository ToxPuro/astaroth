import os


name = "/users/julianlagg/jfksdljfkd"
os.system("rm -f "+name)
os.mkfifo(name)

with open(name, "rb", 0) as fd:
    print("opened")

print("closed")