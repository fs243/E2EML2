import sys
from .logger import logging

def error_message_details(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    filename= exc_tb.tb_frame.f_code.co_filename
    lineno=exc_tb.tb_lineno
    error_msg=str(error)
    error_message=f'error occured in python script name [{filename}] line number [{lineno}] error message [{error_msg}]'
    return error_message

class customeException(Exception):
    def __init__(self,error_message,error_detail:sys):

        super().__init__(error_message)
        
        self.error_message=error_message_details(error=error_message,error_detail=error_detail)

    def str(self):
        return self.error_message
    

