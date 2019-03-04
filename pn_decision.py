import sys
import MeCab
import command as cmd
import fasttext as ft


def text2bow(input):
    m = MeCab.Tagger('-Owakati')
    words = m.parse(input).strip()   
    
    return words

def scoring(prob):
    score = 0.0
    for e in prob.keys():
        score += e * prob[e]
    
    return score

def positive_negative_decision(input, clf):
    prob = {}
    bow = text2bow(input)
    
    estimate = clf.predict_proba(texts=[bow], k=5)[0]

    for e in estimate:
        index = int(e[0][9:-1])
        prob[index] = e[1]

    score = scoring(prob)

    return score

def output(score):
    print("Evaluation Score = " + str(score))
    if score < 1.8:
        print("Result: negative--")
    elif score >= 1.8 and score < 2.6:
        print("Result: negative-")
    elif score >= 2.6 and score < 3.4:
        print("Result: neutral")
    elif score >= 3.4 and score < 4.2:
        print("Result: positive+")
    elif score >= 4.2:
        print("Result: positive++")
    else:
        print("error")
        sys.exit()

def main(model):
    str = input()

    if str == "exit":
        print("bye")
        sys.exit(0)

    score = positive_negative_decision(str, model)
    output(score)

if __name__ == '__main__':
    argvs = sys.argv

    if len(argvs) < 2:
        sys.exit(0)

    model = ft.load_model(argvs[1])
    main(model)

    
    
