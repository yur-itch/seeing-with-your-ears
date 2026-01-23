def generate(n):
    curve = [(0, 0)]
    for i in range(n):
        curve = extend(curve)
    return curve

def extend(curve):
    side_len = int(len(curve) ** 0.5)
    rot90 = rotate90(curve)
    rot270 = rotate270(curve)
    top_left = curve
    top_right = shift(curve, side_len, 0)
    bottom_left = shift(rot90, 0, side_len)[::-1]
    bottom_right = shift(rot270, side_len, side_len)[::-1]
    return bottom_left + top_left + top_right + bottom_right

def rotate90(curve):
    side_len = int(len(curve) ** 0.5)
    return [(side_len-1-y, x) for (x, y) in curve]

def rotate270(curve):
    side_len = int(len(curve) ** 0.5)
    return [(y, side_len-1-x) for (x, y) in curve]

def shift(curve, x, y):
    return [(i + x, j + y) for (i, j) in curve]

def visualize(curve):
    side_len = int(len(curve) ** 0.5)
    output = [[False
               for j in range(side_len * 2 + 1)]
               for i in range(side_len * 2 + 1)]
    output[curve[0][1] * 2][curve[0][0] * 2] = True
    for i in range(1, len(curve)):
        x1, y1 = curve[i - 1]
        x2, y2 = curve[i]
        output[y2 * 2][x2 * 2] = True
        output[y1 + y2][x1 + x2] = True
        
    for i in output:
        for j in i:
            print('#' if j else ' ', end='')
        print()

if __name__ == "__main__":
    visualize(generate(5))