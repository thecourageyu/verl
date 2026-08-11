from datetime import datetime
import asyncio

async def worker(name):
    print(f"{name} start")
    await asyncio.sleep(2)
    print(f"{name} done")

async def main():
    loop = asyncio.get_running_loop()

    start = datetime.now()
    await worker("A")
    await worker("B")
    await worker("C")
    end = datetime.now()

    print(f"{end - start} (sequential, start = {start}; end = {end})")


    start = datetime.now()
    task1 = loop.create_task(worker("A"))
    task2 = loop.create_task(worker("B"))
    task3 = loop.create_task(worker("C"))

    await asyncio.gather(task1, task2, task3)

    end = datetime.now()
    print(f"{end - start} (create task, start = {start}; end = {end})")

asyncio.run(main())