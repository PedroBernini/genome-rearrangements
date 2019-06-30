#!/usr/bin/python2.5
from Tkinter import *
import math
import threading
import time
import tkFont

def get_leftmost(cycle) :
    return cycle[0]

def reorder_cycle(cycle) :
    max = -1
    ind = -1
    for count in range(len(cycle)) :
        if cycle[count] > max :
            ind = count
            max = cycle[count]
    newList = []
    for count in range(len(cycle)) :
        newList.append(cycle[(count + ind) % len(cycle)])
    return newList


# Recebe uma string de numeros entre c(0 e n) ou l(1 e n+1) 
# devolve um ciclo padrao segundo Bafna1998
# kind = 0: Circular cycle
# kind = 1: Linear cycle
def get_cycle(cycle_string, kind = 0) :
    cycle_array_string = cycle_string.split(" ")

    cycle = []
    for el in cycle_array_string :
        int_el = int(el)
        if not kind :
            int_el = int_el + 1
        cycle.append(int_el)
    cycle = reorder_cycle(cycle)
    return cycle


def string_to_cycles(input, kind = 0) :
    cycles_str = re.sub("\[3\]","",input)
    cycles_str = cycles_str.rstrip()
    cycles_str = cycles_str[1:-1]
    cycles_str = cycles_str.split(")(")
    cycles = []
    for cycle_str in cycles_str :
        if cycle_str :
            cycle = get_cycle(cycle_str.rstrip(), kind)
            cycles.append(cycle)
    cycles.sort(key=get_leftmost,reverse=1)
    return cycles

class Draw(threading.Thread) :
    def __init__(self, canvas, cycles) :
        threading.Thread.__init__(self)
        self.cycles = cycles
        self.canvas = canvas

    def draw_indices(self, size, delay_y = 100, delay_x = 0) :
        helv12 = tkFont.Font( family="Helvetica",size=12, weight="bold" )
        y = 100*(delay_y + 1)
        for i in range(0,size+1) :
            x = delay_x + 40*i + 20
            self.canvas.create_text(x,y,text=str(i+1), anchor = 's', font=helv12)

    def draw_canvas(self, cycles, delay_y = 100, delay_x = 0) :
        color = ["red", "black", "blue", "green", "tan","gray", "yellow", "black", "brown", "orange", "black", "black","black","black"]
        #color = ["black", "red", "blue", "yellow", "green", "tan", "gray", "orange", "brown"]
#        color = ["black", "black", "black", "black", "black", "black", "black", "black", "black"]
        median = 100*(delay_y + 1)
        graph_counter = -1
        for cycle in cycles :
            if len(cycles) < 10 :
                graph_counter = graph_counter + 1
            size = len(cycle)
            for i in range(size) :
                from_node = cycle[i]
                to_node   = cycle[(i+1)%size]
                # next_node = cycle[(i+2)%size]

                x1 = abs(from_node) * 40 - 30 + delay_x
                if from_node < 0 :
                    x1 = abs(from_node) * 40 - 10 + delay_x

                x2 = abs(to_node)   * 40  -10 + delay_x
                x3 = x2 - 20
                if to_node < 0 :
                    x2 = abs(to_node)   * 40  -30 + delay_x                

                    x3 = x2 + 20



                y1 = median - 4*math.sqrt(abs(x2-x1))
                y2 = median + 4*math.sqrt(abs(x2-x1))
                self.canvas.create_arc(x1,y1,x2,y2,start=0,extent=180,fill="green",
                                       style="arc", outline = color[graph_counter])

                self.canvas.create_line(x2, median, x3,median, arrow = "last", 
                                        fill = color[graph_counter])

    def print_canvas(self) :
        #psfile = open('%s.ps' % utils.cycles_to_string(self.cycles), 'w')
        psfile = open("xxx.ps",'w')
        psfile.write(self.canvas.postscript(fontmap='fontMap',
                                            colormap='colorMap',
                                            pageanchor='nw',
                                            y='0.c',
                                            x='0.c'))
        psfile.close()

    def run(self) :
        for count in range(0,len(self.cycles)) :
            cycles = self.cycles[count]
            size = 0
            for cycle in cycles :
                size = size + len(cycle)
            size = size - 1

            if count < 8 :
                self.draw_indices(size, count)
                self.draw_canvas(cycles, count)
            else :
                self.draw_indices(size, count % 8 , delay_x = 410)
                self.draw_canvas(cycles, count % 8, delay_x = 410)

        time.sleep(2)
        self.print_canvas()

cycles = []

for count in range(1, len(sys.argv)) :
    arg = eval(sys.argv[count])
    cycles.append(arg)
    #cycles.append(string_to_cycles(arg,kind = 1))

height = 1000
canvas = Canvas(width=1000, height=height, bg='white')
canvas.pack(expand="YES", fill=BOTH)
draw = Draw(canvas, cycles)
draw.start()
mainloop()
