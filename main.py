# -*- coding: utf-8 -*-
"""PyCharm / uvicorn 便捷入口。

真正的应用在 ``app/main.py``；这里只做一层转发，方便 IDE 直接运行本文件
（等价于 ``uvicorn app.main:app --reload``）。
"""
from app.main import app  # noqa: F401  (uvicorn main:app 可直接使用)

if __name__ == "__main__":
    import uvicorn

    from app.config import settings

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
