def print_coeff(i,j):
    prefix = ""
    if i*j < 0:
        prefix = "-"
    print(f" =  {prefix}DER5_{abs(i)}*DER1_{abs(j)},",end="")
def print_entry(k1,k2,i,j,k):
    if k == k1:
        print(f"[{i}]",end="")
    elif k == k2:
        print(f"[{j}]",end="")
    else:
        print("[0]",end="")

def main():
    names = ["z","y","x"]
    for k1 in range(3):
        for k2 in range(3):
            if k1 == k2: 
                continue
            print(f"Stencil der5{names[k1]}1{names[k2]} {{")
            for i in [1,2,3]:
                for j in [1,2,3]:
                    print_entry(k1,k2,i,j,0)
                    print_entry(k1,k2,i,j,1)
                    print_entry(k1,k2,i,j,2)
                    print_coeff(i,j)
                    print("")
                    print_entry(k1,k2,i,-j,0)
                    print_entry(k1,k2,i,-j,1)
                    print_entry(k1,k2,i,-j,2)
                    print_coeff(i,-j)
                    print("")
                for j in [1,2,3]:
                    print_entry(k1,k2,-i,j,0)
                    print_entry(k1,k2,-i,j,1)
                    print_entry(k1,k2,-i,j,2)
                    print_coeff(-i,j)
                    print("")
                    print_entry(k1,k2,-i,-j,0)
                    print_entry(k1,k2,-i,-j,1)
                    print_entry(k1,k2,-i,-j,2)
                    print_coeff(-i,-j)
                    print("")
            print("}\n\n")



if __name__ == '__main__':
        main()
