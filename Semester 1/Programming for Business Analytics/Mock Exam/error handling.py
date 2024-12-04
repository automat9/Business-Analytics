def x(a,b):
    try:
        print(a / b)
    except ZeroDivisionError:
        print("Error, the second number is zero")
    except NameError:
        print('Undefined value')
    except SyntaxError:
        print('Syntax Error')
    except ValueError:
        print(' Invalid value.') 
    except TypeError:
        print('Invalid operation between data types.')
    except IndexError:
        print('Accessing an invalid index in a list.')
    except KeyError:
        print('Missing key in a dictionary.')
    except FileNotFoundError:
        print('File not found.')
    except IndentationError:
        print('Improper indentation.')
    except AttributeError:
        print('Accessing a non-existent attribute.')
    
x(2,4)
