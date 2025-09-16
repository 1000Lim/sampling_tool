import sys
import time
from math import ceil, sqrt
import random
import numpy as np
from multiprocessing import Semaphore
import pickle
import cv2
import gc
from datetime import datetime
import json

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue

number_of_task = 0
current_task = 0
sm = Semaphore(1)

def debug_print(debug, string):
    if debug:
        print(string)


def dict_to_pretty_str(_dict):
    return json.dumps(_dict, indent=4)


def pretty_args(args):
    arg_str = ''
    for arg in vars(args):
        temp = '%-20s   %-30s' % (arg, str(getattr(args, arg)))
        arg_str += temp + '\n'
    return arg_str[:-1]


def print_args(args):
    print('=============== Training Configuration ================')
    print(pretty_args(args))
    print('=======================================================')


def vbr(value, basis, ratio):
    """
    When the ratio approaches 0, it goes to the basis, and when ratio approaches 1, it goes to the value.
    :param value: Value when ratio is 1.
    :param basis: Value when ratio is 0.
    :param ratio: Control ratio between value and basis.
    :return:
    """
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError('The value of ratio must be between 0 and 1.')
    return value + (basis - value) * (1.0 - ratio)


def split_boxes(obj_list):
    x_list = set()
    y_list = set()

    for obj in obj_list:
        x_list.add(obj['x1'])
        x_list.add(obj['x2'])
        y_list.add(obj['y1'])
        y_list.add(obj['y2'])

    x_list = list(x_list)
    y_list = list(y_list)

    x_list.sort()
    y_list.sort()

    boxes = list()
    for x_idx in range(len(x_list) - 1):
        for y_idx in range(len(y_list) - 1):
            boxes.append(
                {
                    'x1': x_list[x_idx],
                    'x2': x_list[x_idx + 1],
                    'y1': y_list[y_idx],
                    'y2': y_list[y_idx + 1],
                }
            )

    return boxes


def is_box_in_box(box_in, box_out):
    if box_out['x1'] <= box_in['x1'] and box_out['x2'] >= box_in['x2'] and \
       box_out['y1'] <= box_in['y1'] and box_out['y2'] >= box_in['y2']:
        return True
    else:
        return False


def box_draw(image, box, color, thickness=1, line_type=cv2.LINE_4):
    cv2.rectangle(image, (box['x1'], box['y1']), (box['x2'], box['y2']), color, thickness, line_type)


def box_same(box1, box2):
    if box1['x1'] == box2['x1'] and box1['x2'] == box2['x2'] and \
       box1['y1'] == box2['y1'] and box1['y2'] == box2['y2']:
        return True
    else:
        return False


def box_remain_area(basis_box, other_boxes, remove_basis_box_from_other_boxes=False):
    """
    Calculate remain area of basis_box which not overlapped by other_boxes.
    :param basis_box: Basis box to calculate remain area.
    :param other_boxes: Other boxes which can be overlapped to basis box.
    :param remove_basis_box_from_other_boxes: Remove basis box from other_boxes if basis box in other_boxes.
    :return: Remain value
    """
    overlapped_boxes = list()
    for box in other_boxes:
        if box_intersect(basis_box, box) != 0 and \
           not (remove_basis_box_from_other_boxes and box_same(basis_box, box)):
            overlapped_boxes.append(box)

    to_split_temp = [basis_box]
    to_split_temp.extend(overlapped_boxes)
    boxes_split = split_boxes(to_split_temp)

    remain = box_area(basis_box)
    for box_s in boxes_split:
        if is_box_in_box(box_s, basis_box):
            for box in overlapped_boxes:
                if is_box_in_box(box_s, box):
                    remain -= box_area(box_s)
                    break

    return remain


def many_box_intersection(box_list, boxes_split=None):
    """
    Calculate sum of intersected area which area intersected by at least two boxes.
    :param box_list: Box list of dict which contains 'x1', 'x2', 'y1', 'y2' as keys.
    :param boxes_split: Boxes which split by obj_list.
    :return: Intersection value
    """
    if boxes_split is None:
        boxes_split = split_boxes(box_list)

    intersect = 0
    for box_s in boxes_split:
        inter_count = 0
        for box in box_list:
            if is_box_in_box(box_s, box):
                inter_count += 1
        if inter_count > 1:
            intersect += box_area(box_s)

    return intersect


def many_box_union(box_list, boxes_split=None):
    """
    Calculate union of all boxes.
    :param box_list: Box list of dict which contains 'x1', 'x2', 'y1', 'y2' as keys.
    :param boxes_split: Boxes which split by obj_list.
    :return: Union value
    """
    if boxes_split is None:
        boxes_split = split_boxes(box_list)

    union = 0
    for box_s in boxes_split:
        for box in box_list:
            if is_box_in_box(box_s, box):
                union += box_area(box_s)
                break

    return union


def box_intersect(box1, box2):
    left = max(box1['x1'], box2['x1'])
    right = min(box1['x2'], box2['x2'])
    top = max(box1['y1'], box2['y1'])
    bottom = min(box1['y2'], box2['y2'])
    if left >= right or top >= bottom:
        return 0
    else:
        return (right - left) * (bottom - top)


def box_area(box):
    return (box['x2'] - box['x1']) * (box['y2'] - box['y1'])


def box_union(box1, box2):
    return box_area(box1) + box_area(box2) - \
           box_intersect(box1, box2)


def box_intersection_over_union(box1, box2):
    return (box_intersect(box1, box2)) / \
           box_union(box1, box2)


def save_obj(obj, path):
    # disable garbage collector
    gc.disable()

    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    # enable garbage collector again
    gc.enable()


def load_obj(path):
    # disable garbage collector
    gc.disable()

    with open(path, 'rb') as f:
        data = pickle.load(f)

    # enable garbage collector again
    gc.enable()

    return data


def chance(p=1.0):
    """
    Return True with probability of p.
    :param p: Probability
    :return:
    """
    r = random_float(0, 1)
    if r <= p:
        return True
    else:
        return False


def random_float(f_min=0.0, f_max=1.0):
    if f_min > f_max:
        raise ValueError('f_min must be smaller than f_max')
    return float(random.random() * (f_max - f_min) + f_min)


def random_normal_float(f_min, f_max, scale=1.5):
    """
    Return normal value as float
    :param f_min: Minimum value
    :param f_max: Maximum value
    :param scale: Higher scale make the return value uniform
    :return:
    """
    if f_min > f_max:
        raise ValueError('f_min(%f) must be smaller than f_max(%f)' % (f_min, f_max))
    norm = (min(max(np.random.normal(5., scale), 0.), 10.) - 5.) / 5.  # [-1.,1.] Normalized value
    norm *= (f_max - f_min) / 2.  # scale
    norm += (f_max + f_min) / 2.  # bias
    return min(max(norm, f_min), f_max)


def random_int(i_min, i_max, num=1):
    if i_min > i_max:
        raise ValueError('i_min must be smaller or equal than i_max')
    if num == 1:
        return int(random.randrange(int(i_min), int(i_max) + 1))
    elif num > 1:
        return [int(random.randrange(int(i_min), int(i_max) + 1)) for _ in range(num)]
    else:
        raise ValueError('num must be larger or equal than 1.')


def random_normal_int(i_min, i_max, scale=1.8):
    if i_min > i_max:
        raise ValueError('i_min(%d) must be smaller than i_max(%d)' % (i_min, i_max))
    norm = (min(max(np.random.normal(5., scale), 0.), 10.) - 5.) / 5.  # [-1.,1.] Normalized value
    norm *= (i_max - i_min) / 2.  # scale
    norm += (i_max + i_min) / 2.  # bias
    return int(min(max(norm, i_min), i_max))


def print_function(string):
    sys.stdout.write('\r' + string)
    sys.stdout.flush()


def divide_list(task_list, n_worker, mode='block'):
    """
    Divide tasks per worker
    :param task_list: List of whole tasks
    :param n_worker: Number of workers
    :param mode: 'block' or 'sequential'
        example) task_list = [1,2,3,4,5,6,7,8,9,10]
                    n_worker = 4
                    mode = 'block'
                    return = [[1,2,3,4], [5,6,7], [8,9,10]]

                    n_worker = 4
                    mode = 'sequential'
                    return = [[1,5,9], [2,6,10], [3,7], [4,8]]

    :return: Division list
    """
    division = list()
    if mode == 'block':
        n_list = len(task_list)
        division_size = n_list//n_worker
        remain = n_list % n_worker
        start = 0
        for t in range(n_worker):
            cur_size = division_size + (1 if t < remain else 0)
            division.append(task_list[start:start + cur_size])
            start += cur_size
    elif mode == 'sequential':
        for t in range(n_worker):
            division.append(task_list[t::n_worker])
    else:
        raise ValueError('mode must be block or sequential')

    return division


class Timer:

    def __init__(self, display_every=0, inline_print=True, as_progress_notifier=True, time_queue_size=1000):
        """
        Thread safe Timer class.
        If set as_progress_notifier = True, then it will be use to check progress of some processes.
        If not it will be use to simple timer.
        :param as_progress_notifier:
        """
        self.whole_number_of_data = 0
        self.start_progress = 0
        self.current_progress = 0
        self.prev_progress = 0
        self.time_moving_average = 0
        self.elapsed_time = 0
        self.start_time = 0
        self.remain_time = 0
        self.tick_start_time = 0
        self.is_progress_notifier = as_progress_notifier
        self.timer_ready = False
        self.print_fn = self.timer_print
        self.time_queue = queue.Queue()
        self.time_queue_size = time_queue_size
        self.locker = Semaphore(1)
        self.time_sum = 0
        self.inline_print = inline_print
        self.display_every = display_every
        self.time_average = 0

    def now(self):
        return time.time()

    def get_remain_time(self, format=None):
        if format is not None:
            return time.strftime(format, self.get_remain_time())
        else:
            return time.gmtime(self.remain_time)

    def remain_hour(self):
        return int(self.remain_time / 3600)

    def remain_minute(self):
        return int(self.remain_time / 60) % 60

    def remain_second(self):
        return int(self.remain_time % 60)

    def progress_in_percent(self):
        return float(self.current_progress) / float(self.whole_number_of_data) * 100

    def left_time(self):
        return [self.remain_hour(), self.remain_minute(), self.remain_second()]

    def timer_print(self, *args, **kwargs):
        """
        Print function for timer
        :param args: Arguments for printing out string which relative to 'format' of kwargs
        :param kwargs: Argument which key is 'format' as format of output string
        :return:
        """
        if 'format' not in kwargs:
            return 'Timer : [%d/%d][%.2f%%][%dh %dm %ds][%.2f iter/sec][%.2f sec/%diter]' % (
                self.current_progress, self.whole_number_of_data,
                self.progress_in_percent(), self.remain_hour(), self.remain_minute(), self.remain_second(),
                -1 if self.time_average == 0 else (1. / self.time_average), self.time_average * self.display_every, self.display_every
            )
        else:
            return kwargs['format'] % args

    def print_out(self, output_str):
        if self.inline_print:
            sys.stdout.write('\r' + output_str)
            sys.stdout.flush()
        else:
            print(output_str)

    def start(self, number_of_data=1, start_progress=None):
        if self.is_progress_notifier:
            self.start_progress = 0 if start_progress is None else start_progress
            self.current_progress = 0 if start_progress is None else start_progress
            self.time_queue.queue.clear()
        self.start_time = time.time()
        self.whole_number_of_data = number_of_data
        self.tick_start_time = time.time()
        self.time_sum = 0
        self.timer_ready = True

    def tick_timer(self, *args, **kwargs):
        self.locker.acquire()
        if not self.timer_ready:
            raise AttributeError('Need to initialize timer by start().')
        if not self.is_progress_notifier:
            raise AttributeError('You should set as_progress_notifier to True if you want to use tick_timer().')

        self.current_progress += 1

        if self.display_every == 0 or self.current_progress % self.display_every == 0 or self.current_progress == self.whole_number_of_data:
            t_time = time.time() - self.tick_start_time
            self.time_queue.put(t_time)
            self.time_sum += t_time
            if self.time_queue.qsize() > self.time_queue_size:
                self.time_sum -= self.time_queue.get()

            self.time_average = float(self.time_sum) / float(self.time_queue.qsize()) / (self.current_progress - self.prev_progress)

            self.remain_time = (self.whole_number_of_data - self.current_progress) * self.time_average

            if self.display_every != 0 and self.print_fn is not None:
                self.print_out(self.print_fn(*args, **kwargs))

            self.tick_start_time = time.time()
            self.elapsed_time = time.time() - self.start_time

            self.prev_progress = self.current_progress

        self.locker.release()

        if self.whole_number_of_data == self.current_progress:
            return True
        else:
            return False

    def check(self, *args, **kwargs):
        self.locker.acquire()
        if self.is_progress_notifier:
            raise AttributeError('You should set as_progress_notifier to False if you want to use check().')
        self.elapsed_time = time.time() - self.tick_start_time

        if self.print_fn is not None:
            self.print_out(self.print_fn(*args, **kwargs))

        self.start(self.whole_number_of_data)
        self.locker.release()


# TODO : Add many timers
task_timer = Timer(time_queue_size=5000)


def start_task(number_of_tasks, initial_task=0, display_every=1, timer_queue_size=-1):
    global task_timer
    if timer_queue_size == -1:
        task_timer.time_queue_size = int(number_of_tasks / 10)
    else:
        task_timer.time_queue_size = timer_queue_size
    if task_timer.time_queue_size == 0:
        task_timer.time_queue_size = 1
    task_timer.start(number_of_tasks, initial_task)
    task_timer.display_every = display_every


def print_task(*args, **kwargs):
    global task_timer
    task_timer.tick_timer(*args, **kwargs)


def print_with_time(text):
    print('[%s] %s' % (datetime.now().strftime('%Y.%m.%d %H:%M:%S'), text))
