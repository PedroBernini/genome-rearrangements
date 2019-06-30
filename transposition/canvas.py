import sys
from tkinter import *
import tkinter.font as tkFont
import math
import threading
import time
#import tkFont

class Draw(threading.Thread) :
    def __init__(self, canvas, permutation) :
        threading.Thread.__init__(self)
        self.permutation = permutation
        
        black_indices = {}
        previous = 0
        for el in permutation[0] + [(len(permutation[0])+1)]  :
            el_tuple = (-el, el)
            black_indices[previous] = el_tuple[0]
            black_indices[el_tuple[0]] = previous
            previous = el_tuple[1]
        self.black_indices = black_indices

        self.canvas = canvas

    def create_graph(self, permutation) :
        graph = [[0, 10, False]]

        #canvas.create_line(10,200,30,200,arrow = "first")

        count = 1
        for item in permutation :
            graph.append([-item, count*40 - 10, False])
            graph.append([ item, count*40 + 10, False])
            count = count + 1
            #canvas.create_line((count-1)*40 + 10,200,count*40 - 10,200, arrow = "first")

        graph.append([-(len(permutation) + 1), count*40-10, False])
        return graph

    def draw_indices(self, size, delay_y = 100, delay_x = 0) :
        helv12 = tkFont.Font( family="Helvetica",size=12, weight="bold" )
        y = 130*(delay_y + 1)
        for i in range(0,size+1) :
            x = delay_x + 40*i + 20
            self.canvas.create_text(x,y,text=str(i+1), anchor = 's', font=helv12)

    def draw_labels(self, permutation, delay_y = 100, delay_x = 0) :
        helv10 = tkFont.Font( family="Helvetica",size=10 )
        y = 130*(delay_y+1)+2
        size = len(permutation)
        x = delay_x + 10
        self.canvas.create_text(x,y,text="0",anchor = 'n', font=helv10)

        for i in range(0,size) :
            x = 40*i+40 + delay_x
            self.canvas.create_text(x-10,y,text=str(-permutation[i]),anchor='n', font=helv10)
            self.canvas.create_text(x+10,y,text=str(permutation[i]),anchor='n', font= helv10)
        self.canvas.create_text(40*(size) + 30 + delay_x,y,text=str(-(size+1)),anchor='n',font=helv10)

    def draw_canvas(self, graph, delay_y = 100, delay_x = 0) :
        color = ["blue", "black", "red", "green", "gray", "yellow", "tan", "black", "black", "black","black","black"]
#        color = ["black", "black", "black", "black", "black", "black", "black", "black", "black", "black","black","black"]
        
        median = 130*(delay_y + 1)
        graph_counter = -1
        reverse_graph = list(graph)
        reverse_graph.reverse()
        for x in reverse_graph :

            ## Adjusting from_node

            first_search = -(x[0] + 1)           
            from_node = 0
            
            search_node = 0
            while not from_node :
                if  graph[search_node][0] == first_search :
                    from_node = graph[search_node]
                else :
                    search_node = search_node + 1


            if not from_node[2] :
                graph_counter = graph_counter + 1

            while not from_node[2]  :
                from_node[2] = True
                
                to_node = 0
                search_node = 0
                while not to_node :
                    if  graph[search_node][0] == -(from_node[0] + 1) :
                        to_node = graph[search_node]
                    else :
                        search_node = search_node + 1

                to_node[2] = True

                next_node   = 0
                if search_node % 2 == 0 :
                    next_node = graph[search_node + 1]
                else :
                    next_node = graph[search_node - 1]


                x1 = from_node[1] + delay_x
                x2 = to_node[1] + delay_x
                x3 = next_node[1] + delay_x
                y1 = median - 4*math.sqrt(abs(x2-x1))
                y2 = median + 4*math.sqrt(abs(x2-x1))
                
                ## Gray edges
                self.canvas.create_arc(x1,y1,x2,y2,start=0,extent=180,fill="green",
                                       style="arc", outline = color[graph_counter])

                ## Black edges
                self.canvas.create_line(x2, median, x3,median, arrow = "last", 
                                        fill = color[graph_counter])
                from_node = next_node

    def print_canvas(self) :
        psfile = open('xxx.ps', 'w')
        psfile.write(self.canvas.postscript(fontmap='fontMap',
                                            colormap='colorMap',
                                            pageanchor='nw',
                                            y='0.c',
                                            x='0.c'))
        psfile.close()


    def run(self) :
        for count in range(0,len(self.permutation)) :
            if count < 10 :
                graph = self.create_graph(self.permutation[count])
                self.draw_indices(len(self.permutation[count]), count)
                self.draw_labels(self.permutation[count],count)
                self.draw_canvas(graph, count)
            else :
                graph = self.create_graph(self.permutation[count])
                self.draw_indices(len(self.permutation[count]), count % 5, delay_x = 410)
                self.draw_labels(self.permutation[count],count % 5, delay_x = 410)
                self.draw_canvas(graph, count % 5, delay_x = 410)
                                                                                

        time.sleep(2)
        self.print_canvas()



permutation = []
for perm in range(1, len(sys.argv)) :
    str_permutation = sys.argv[perm].split(",")
    permutation.append([])
    for el in str_permutation :
        permutation[perm-1].append(int(el))

height = 150*len(permutation)
canvas = Canvas(width=1000, height=height, bg='white')
canvas.pack(expand="YES", fill=BOTH)
draw = Draw(canvas, permutation)
draw.start()
mainloop()

