#!/usr/bin/env python3

import time

def duration_estimate( iterations_past, iterations_total, current_duration ):
    time_left = time.gmtime((iterations_total - iterations_past) * current_duration)
    return ''.join([
     '{} h '.format( time_left.tm_hour ) if time_left.tm_hour > 0 else '',
     '{} mn'.format( time_left.tm_min ) ])

