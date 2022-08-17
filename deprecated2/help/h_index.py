def solution(citations):
    citations = sorted(citations)
    len_c = len(citations)
    papers = list(range(1, len_c + 1))[::-1]
    print(citations)
    print(papers)

    answer = 0
    for n, c in enumerate(citations, start=0):
        p = len_c - n
        if c <= p:
            answer = c
        else:
            break
        print(c, p, c <= p)
    print(answer)

    return answer


c = [3, 0, 6, 1, 5]

solution(c)
