import sys
from session import PRFSession
import appnope


def main():
    initials = sys.argv[1]
    index_number = int(sys.argv[2])

    appnope.nope()

    ts = PRFSession(subject_initials=initials, index_number=index_number, tracker_on=False)
    ts.run()

if __name__ == '__main__':
    main()
