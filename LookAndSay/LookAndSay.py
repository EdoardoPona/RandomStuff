""" Playing around with the Look and Say sequence: https://en.wikipedia.org/wiki/Look-and-say_sequence """
import matplotlib.pylab as plt


def next_string(current_string):
    """ we iterate along the current string and append the numbers to another string until we encounter a different one, at that point we append the
    length of the partial string and one of its characters to the output """
    output = ''
    partial_string = ''     # this will hold the numbers while they are the same
    partial_string += current_string[0]

    for i in range(len(current_string)-1):
        if current_string[i+1] == partial_string[-1]:
            partial_string += current_string[i+1]
        else:
            output += str(len(partial_string)) + partial_string[-1]
            partial_string = current_string[i+1]

    output += str(len(partial_string)) + partial_string[-1]

    return output


def compute_strings(initial_string='1', n=100, difference_grade=15):
    ''' as well as calculating the lines up to n this function calculates the difference between the length of two consecutive
    lines, and the difference between two consecutive differences, and so on until difference_grade '''
    strings = [initial_string]
    lengths = [len(initial_string)]

    list_of_differences = [[0] for i in range(difference_grade)]        # difference grade 0 is the lengths
    list_of_differences[0] = [0]

    log_files = []          # calculating beyond line 70 takes ages on my computer so I'm saving the lines 
    for i in range(difference_grade):
        log_files.append(open('differences'+str(i)+'.txt', mode='w'))

    for i in range(n):
        strings.append(next_string(strings[-1]))
        lengths.append(len(strings[-1]))

        list_of_differences[0].append(len(strings[-1]))
        log_files[0].write(str(list_of_differences[0][-1]) + '\n')

        for j in range(len(list_of_differences)-1):
            list_of_differences[j+1].append(abs(list_of_differences[j][-1] - list_of_differences[j][-2]))
            log_files[j+1].write(str(list_of_differences[j+1][-1]) + '\n')

        print(i, list_of_differences[0][-1])

    return strings, list_of_differences


def read_files(n=15):
    """ reading the saved files """
    list_of_differences = [[0] for i in range(n)]

    for i in range(n):
        with open('differences'+str(i)+'.txt', 'r') as file:
            for line in file:
                list_of_differences[i].append(int(line.rstrip('\n')))

    return list_of_differences


def plot_files(n=15):
    """ plot the contents of the saved files """
    lod = read_files(n)

    for i in range(len(lod)):
        plt.plot(range(len(lod[i])), lod[i])

    plt.show()


def conway_polynomial(n=100):
    """ Conway's polynomial has as only real positive solution increase in line length as the number of lines tends to infinite
    plotting this function is very cool within [-1,15;+1.18], beyond that is explodes """
    
    xs = []
    ys = []
    for i in range(n):
        x = (i - n/2) / 50
        xs.append(x)

        y = x**71 - x**69 - 2*x**68 - x**67 + 2*x**66 + 2*x**65 + x**64 - x**63 - x**62 - x**61 - x**60 - x**59 + 2*x**58 + 5*x**57 + 3*x**56 +\
            - 2*x**55 - 10*x**54 - 3*x**53 - 2*x**52 + 6*x**51 + 6*x**50 + x**49 + 9*x**48 - 3*x**47 - 7*x**46 - 8*x**45 - 8*x**44 + 10*x**43 +\
            + 6*x**42 + 8*x**41 - 5*x**40 - 12*x**39 + 7*x**38 - 7*x**37 + 7*x**36 + x**35 - 3*x**34 + 10*x**33 + x**32 - 6*x**31 - 2*x**30 +\
            - 10*x**29 - 3*x**28 + 2*x**27 + 9*x**26 - 3*x**25 + 14*x**24 - 8*x**23 - 7*x**21 + 9*x**20 + 3*x**19 - 4*x**18 - 10*x**17 - 7*x**16 +\
            + 12*x**15 + 7*x**14 + 2*x**13 - 12*x**12 - 4*x**11 - 2*x**10 + 5*x**9 + x**7 - 7*x**6 + 7*x**5 - 4*x**4 + 12*x**3 - 6*x**2 + 3*x - 6

        ys.append(y)

    return xs, ys

# _, lod = compute_strings(n=70)
# xs, ys = conway_polynomial(n=118)


