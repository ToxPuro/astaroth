import argparse
def main():
    argparser = argparse.ArgumentParser(description="Astaroth helper script to gen dlsym calls from FUNC_DEFINES",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("-F", "--file", help="file to be parsed", required=True)
    filename = vars(argparser.parse_args())["file"]
    res = "\tvoid* handle = dlopen(runtime_astaroth_path,RTLD_NOW);\n"
    res += 'if(!handle){fprintf(stderr,"%s","Fatal error was not able to load Astaroth\\n"); exit(EXIT_FAILURE);}\n'
    funcs_added = []
    with open(filename) as file:
        for line in [line for line in file if "FUNC_DEFINE" in line and "#" not in line]:
            parts = [part.strip() for part in line.split(",")]
            #return_type = parts[0]
            func_name = parts[1]
            if func_name in funcs_added:
                continue
            funcs_added.append(func_name)
            #argument_list = parts[2]
            if ("OVERLOADED_FUNC_DEFINE" in line):
                res += f'\t*(void**)(&BASE_FUNC_NAME({func_name})) = dlsym(handle,"{func_name}");\n'
            else:
                res += f'\t*(void**)(&{func_name}) = dlsym(handle,"{func_name}");\n'
                res += f'\tif(!{func_name}) fprintf(stderr,"Astaroth error: was not able to load %s\\n","{func_name}");\n'
    res += "\tdlclose(handle);"
    print(res)
if __name__ == "__main__":
    main()
