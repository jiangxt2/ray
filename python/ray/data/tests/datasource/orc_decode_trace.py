"""Instrument the production stripe reader inside real Ray read workers."""

import os
from dataclasses import fields

import ray
from ray.data._internal.datasource_v2.formats.orc.orc_scanner import OrcScanner


def traced_scanner(scanner, trace):
    class TracedScanner(OrcScanner):
        def create_reader(self):
            reader = super().create_reader()
            original = reader._iter_stripe_tables

            def read(fragment, index, kwargs):
                ray.get(trace.record.remote(index, os.getpid()))
                yield from original(fragment, index, kwargs)

            reader._iter_stripe_tables = read
            return reader

    return TracedScanner(
        **{
            field.name: getattr(scanner, field.name)
            for field in fields(scanner)
            if field.init
        }
    )
