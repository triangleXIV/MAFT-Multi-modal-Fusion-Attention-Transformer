import logging

logging.disable(logging.DEBUG)  # 关闭DEBUG日志的打印

logger = logging.getLogger()

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Setup file handler
fhandler = logging.FileHandler('./my.log')
fhandler.setLevel(logging.INFO)
fhandler.setFormatter(formatter)

# Configure stream handler for the cells
chandler = logging.StreamHandler()
chandler.setLevel(logging.INFO)
chandler.setFormatter(formatter)

# Add both handlers
logger.addHandler(fhandler)
logger.addHandler(chandler)
logger.setLevel(logging.INFO)

# Show the handlers
logger.handlers