import sys
from session import MSSession
import appnope


def main():
    initials = sys.argv[1]
    #initials = raw_input('Your initials: ')
    #run_nr = int(raw_input('Run number: '))
    #scanner = raw_input('Are you in the scanner (y/n)?: ')
    #track_eyes = raw_input('Are you recording gaze (y/n)?: ')
    # if track_eyes == 'y':
        #tracker_on = True
    # elif track_eyes == 'n':
        #tracker_on = False

    # initials = 'tk'
    run = int(sys.argv[2])
    appnope.nope()

    ts = MSSession(subject_initials=initials, index_number=run, tracker_on=True)
    ts.run()

if __name__ == '__main__':
    main()
