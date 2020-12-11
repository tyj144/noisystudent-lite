import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: graph_out.py <RUN_NAME>")
    exit()

run_path = sys.argv[1] 

def parse_accuracies(run_path, type="val"):
    '''
    Get all accuracies over time from a .out file.
    '''
    with open(run_path) as f:
        l = f.read()
        lines = l.split("\n")
        vals = [float(line.split(" ")[-1]) for line in lines if type in line]
        # last line is overall best accuracy
        vals = vals[:-1]
        print(vals)
    return vals

def show_accuracies(accuracies):
    plt.plot(range(len(accuracies)), accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Validation accuracies')
    plt.axis([-5, 105, -0.05, 1])
    plt.show()

student = parse_accuracies("runs/Student_78pct.out")
teacher = parse_accuracies("runs/TeacherTest_63pct.out")
plt.plot(range(len(student)), student, label='Student')
plt.plot(range(len(teacher)), teacher, label='Teacher')
plt.xlabel('Epochs')
plt.ylabel('Validation accuracy')
plt.axis([-5, 105, -0.05, 1])
plt.legend()
plt.title("CUB-200 2010 validation accuracy over time")
plt.show()