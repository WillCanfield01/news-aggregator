"""create shared_tool_links table

Revision ID: 2025_12_21_01
Revises: 
Create Date: 2025-12-21
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "2025_12_21_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "shared_tool_links",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("token", sa.String(length=64), nullable=False),
        sa.Column("tool", sa.String(length=50), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
    )
    op.create_index("ix_shared_tool_links_token", "shared_tool_links", ["token"], unique=True)
    op.create_index("ix_shared_tool_links_tool", "shared_tool_links", ["tool"])
    op.create_index("ix_shared_tool_links_created_at", "shared_tool_links", ["created_at"])
    op.create_index("ix_shared_tool_links_expires_at", "shared_tool_links", ["expires_at"])
    op.create_index("ix_shared_tool_links_user_id", "shared_tool_links", ["user_id"])


def downgrade():
    op.drop_table("shared_tool_links")
