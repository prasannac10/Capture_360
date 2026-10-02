"""Materialize a local folder or an S3 dataset snapshot before training."""
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
import uuid


def materialize_dataset(location, cache_dir, base=None, client=None):
    location = str(location)
    if not location.startswith('s3://'):
        path = Path(location).expanduser()
        if not path.is_absolute():
            path = Path(base or Path.cwd()) / path
        if not path.is_dir():
            raise FileNotFoundError(f'Dataset folder does not exist: {path}')
        return path.resolve()
    parsed = urlsplit(location)
    if not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError('Use s3://bucket/prefix without query parameters')
    prefix = parsed.path.lstrip('/').rstrip('/')
    prefix = prefix + '/' if prefix else ''
    if client is None:
        try:
            import boto3
        except ImportError as error:
            raise ImportError('S3 input requires boto3; install requirements-training.txt') from error
        client = boto3.client('s3')  # Standard AWS credentials/instance role chain.
    objects = []
    for page in client.get_paginator('list_objects_v2').paginate(Bucket=parsed.netloc, Prefix=prefix):
        objects.extend({k: row[k] for k in ('Key', 'Size', 'ETag')}
                       for row in page.get('Contents', []) if not row['Key'].endswith('/'))
    if not objects:
        raise ValueError(f'No dataset objects under {location}')
    objects.sort(key=lambda item: item['Key'])
    uri_hash = hashlib.sha256(location.encode()).hexdigest()[:16]
    snapshot_hash = hashlib.sha256(json.dumps(objects, sort_keys=True).encode()).hexdigest()[:20]
    snapshot = Path(cache_dir).resolve() / uri_hash / snapshot_hash
    root = snapshot / 'data'
    complete = snapshot / 'complete.json'
    destinations = []
    for item in objects:
        if not item['Key'].startswith(prefix):
            raise ValueError('S3 listing returned a key outside the requested prefix')
        relative = item['Key'][len(prefix):]
        parts = PurePosixPath(relative).parts
        if not parts or any(p in ('.', '..') or ':' in p or '\\' in p for p in parts) or relative.startswith('/'):
            raise ValueError(f'Unsafe dataset key: {item["Key"]}')
        target = root.joinpath(*parts).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError('Dataset key escapes local cache')
        destinations.append(target)
    if complete.exists() and all(p.is_file() and p.stat().st_size == row['Size']
                                 for p, row in zip(destinations, objects)):
        return root
    for item, target in zip(objects, destinations):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.partial-' + uuid.uuid4().hex)
        client.download_file(parsed.netloc, item['Key'], str(temporary))
        current = client.head_object(Bucket=parsed.netloc, Key=item['Key'])
        if current['ETag'] != item['ETag'] or temporary.stat().st_size != item['Size']:
            temporary.unlink()
            raise RuntimeError('Dataset changed during download; use an immutable/versioned dataset prefix and retry')
        temporary.replace(target)
    complete.write_text(json.dumps({'uri': location, 'objects': objects}, indent=2), encoding='utf-8')
    return root
