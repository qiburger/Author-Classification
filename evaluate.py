

def get_true_markings(true_result_address):
    '''
        Get the true markings from the training set
    '''

    results = []
    counter = 0
    with open(true_result_address, "r") as fin:
        for line in fin:
            temp = line.rstrip().split('\t')
            if len(temp) != 2:
                raise ValueError("Maybe more than one tab at line " + str(counter))
            counter += 1
            tagger = int(temp[1])
            if tagger == 0 or tagger == 1:
                results.append(tagger)
            else:
                raise ValueError("Something wrong with the 0/1 at line " + str(counter))
    return results


def get_submission_results(submission_file):
    '''
        Get results of a validation file with submission format (0 or 1 per line) to a list
    '''

    results = []
    counter = 0
    with open(submission_file, "r") as fin:
        for line in fin:
            tagger = int(line.rstrip())
            if tagger == 0 or tagger == 1:
                results.append(tagger)
                counter += 1
            else:
                raise ValueError("Something wrong with the 0/1 at line " + str(counter))
    return results


def compare_results(submission_file, true_result_address):
    '''
        Compare a validation file and the true results, where the true results is from the training set.
        Note the submission file would contain the first n predicted markings, 
        which will be compared to the first n paragraphs in training.
    '''


    true_pos = 0
    false_pos = 0
    false_neg = 0
    true_neg = 0

    results = get_submission_results(submission_file)
    true_results = get_true_markings(true_result_address)
    
    if len(results) > len(true_results):
        raise ValueError("length of test longer than true results")
    for i in range(len(results)):
        if true_results[i] == 1 and results[i] == 1:
            true_pos += 1
        elif true_results[i] == 0 and results[i] == 1:
            false_pos += 1
        elif true_results[i] == 1 and results[i] == 0:
            false_neg += 1
        elif true_results[i] == 0 and results[i] == 0:
            true_neg += 1

    out = [true_pos, false_pos, false_neg, true_neg]

    print "True Positive: " + str(true_pos)
    print "False Positive: " + str(false_pos)
    print "False Negative: " + str(false_neg)
    print "True Negative: " + str(true_neg)
    
    return out


def get_delta_analytics():
    '''
        This function analyzes our match scores for all test sets and see
        how big the discrepencies are between the 0 and 1 decisions for reach paragraph.
    '''
    counter = 0
    author_list = []
    non_author_list = []

    delta_list = []
    abs_delta_list = []

    non_author_delta_list = []
    author_delta_list = []

    with open("test_score_details.txt", "r") as fin:
        for line in fin:
            if counter % 3 == 1:
                author_list.append(float(line.rstrip()))
            elif counter % 3 == 2:
                non_author_list.append(float(line.rstrip()))
            counter += 1

    if len(author_list) != len(non_author_list):
        raise ValueError("length different")

    for i in range(len(author_list)):
        diff = author_list[i] - non_author_list[i]
        delta_list.append(diff)
        abs_delta_list.append(abs(diff))

        if diff < 0:
            non_author_delta_list.append(abs(diff))
        elif diff > 0:
            author_delta_list.append(abs(diff))

    print len(delta_list)

    print "First glance for the absolute value of differences: "
    print "min difference is: " + str(min(abs_delta_list))
    print "max difference is: " + str(max(abs_delta_list))
    print "avg difference is: " + str(sum(abs_delta_list) / float(len(abs_delta_list)))

    print "Now, for paragraph with higher non author scores, let's examine the absolute value of difference:"
    print "min difference is: " + str(min(non_author_delta_list))
    print "max difference is: " + str(max(non_author_delta_list))
    print "avg difference is: " + str(sum(non_author_delta_list) / float(len(non_author_delta_list)))

    print "Now, for paragraph with higher author scores, let's examine the absolute value of difference:"
    print "min difference is: " + str(min(author_delta_list))
    print "max difference is: " + str(max(author_delta_list))
    print "avg difference is: " + str(sum(author_delta_list) / float(len(author_delta_list)))


if __name__ == "__main__":
    print compare_results("validation_results.txt", "project_articles_train")
    # get_delta_analytics()



