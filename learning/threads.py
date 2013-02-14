import threading, zipfile
import logging as log
'''
Created on Feb 13, 2013

@author: Felix
'''
log.basicConfig(filename='config.txt')

class threads(threading.Thread):
    def __init__(self, infile, outfile):
        threading.Thread.__init__(self)
        self.infile=infile
        self.outfile=outfile
    
    def run(self):
        f=zipfile.ZipFile(self.outfile, 'w', zipfile.ZIP_DEFLATED)
        f.write(self.infile)
        f.close()
        print 'Finished background zip of: ', self.infile
        log.debug('something that is useful config.txt')
        
background=threads('test.txt', 'test.zip')
background.start()
print 'The main program continues to run in the foreground.'

background.join()
print 'Main program waited until backgroun was done'