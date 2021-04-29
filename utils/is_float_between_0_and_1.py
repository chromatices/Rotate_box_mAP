"""
 check if the number is a float between 0.0 and 1.0
"""


def is_float_between_0_and_1(value : float) -> float:
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False

if __name__ == '__main__':
    print(is_float_between_0_and_1(0.7))
    print(is_float_between_0_and_1(1.7))