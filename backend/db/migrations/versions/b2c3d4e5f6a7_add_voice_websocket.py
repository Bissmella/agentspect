"""add voice_websocket to protocoldb enum

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-10-11 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLAlchemy stores enum *names* as labels, so the new label is uppercase.
    # PG 12+ allows ADD VALUE inside a transaction as long as it is not used in
    # the same one; this migration only adds it. IF NOT EXISTS keeps it idempotent.
    op.execute("ALTER TYPE protocoldb ADD VALUE IF NOT EXISTS 'VOICE_WEBSOCKET'")


def downgrade() -> None:
    # Postgres cannot drop a value from an enum type, so this is a no-op.
    pass
