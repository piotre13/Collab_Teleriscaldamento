import asyncio
import aiomas
import click
import sys
import os
from . import util


@click.command()
@click.option('--start-date', required=True, callback=util.validate_start_date,
              help='Start date for the simulation (ISO-8601 compliant, e.g.: '
                   '2010-03-27T00:00:00+01:00')
@click.argument('addr', metavar='HOST:PORT', callback=util.validate_addr)
def main(addr, start_date):
    try:
        print('starting a new container as subprocess: %s \n' % os.getpid())
        container_kwargs = util.get_container_kwargs(start_date)
        task = aiomas.subproc.start(addr, **container_kwargs)
        aiomas.run(until=task)

    finally:
        print('closing the event loop of subprocess : %s \n' % os.getpid())
        asyncio.get_event_loop().close()


if __name__ == '__main__':
    main()
