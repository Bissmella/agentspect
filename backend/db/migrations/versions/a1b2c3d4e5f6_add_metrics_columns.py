"""add target_constraint and recovery_quality to scenarios

Revision ID: a1b2c3d4e5f6
Revises: 88865c985b72
Create Date: 2026-06-17 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '88865c985b72'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('scenarios', sa.Column('target_constraint', sa.String(length=1024), nullable=True))
    op.add_column('scenarios', sa.Column('recovery_quality', sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column('scenarios', 'recovery_quality')
    op.drop_column('scenarios', 'target_constraint')
