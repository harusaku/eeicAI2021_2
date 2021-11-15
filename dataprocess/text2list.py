def text2list(filename):
    f = open(filename, 'r', encoding='UTF-8')
    ret = []
    for line in f:
        ret.append(line.rstrip())
    f.close()
    return ret