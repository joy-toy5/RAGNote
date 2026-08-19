"""增加 M2 原件、文档版本与稳定 chunk 事实表。

Revision ID: 0002_index_contract
Revises: 0001_legacy
Create Date: 2026-08-17
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

revision: str = "0002_index_contract"
down_revision: str | Sequence[str] | None = "0001_legacy"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ASCII_SHA256 = sa.CHAR(64).with_variant(
    mysql.CHAR(64, collation="ascii_bin"),
    "mysql",
)
ASCII_UUID = sa.CHAR(36).with_variant(
    mysql.CHAR(36, collation="ascii_bin"),
    "mysql",
)
ASCII_USER_ID = sa.String(64).with_variant(
    mysql.VARCHAR(64, collation="ascii_bin"),
    "mysql",
)
ASCII_SOURCE_TYPE = sa.String(32).with_variant(
    mysql.VARCHAR(32, collation="ascii_bin"),
    "mysql",
)
UTF8_DISPLAY_NAME = sa.String(512).with_variant(
    mysql.VARCHAR(512, collation="utf8mb4_bin"),
    "mysql",
)
UTF8_STORAGE_URI = sa.String(1024).with_variant(
    mysql.VARCHAR(1024, collation="utf8mb4_bin"),
    "mysql",
)
ASCII_MEDIA_TYPE = sa.String(255).with_variant(
    mysql.VARCHAR(255, collation="ascii_bin"),
    "mysql",
)
UNSIGNED_INTEGER = sa.Integer().with_variant(mysql.INTEGER(unsigned=True), "mysql")
MYSQL_TABLE_OPTIONS = {
    "mysql_charset": "utf8mb4",
    "mysql_collate": "utf8mb4_bin",
    "mysql_engine": "InnoDB",
}


def upgrade() -> None:
    """创建 M2 索引事实表，不回填或改写 legacy 数据。"""
    op.create_table(
        "content_blobs",
        sa.Column("blob_id", ASCII_SHA256, nullable=False, comment="原始字节 SHA-256"),
        sa.Column("byte_size", sa.BigInteger(), nullable=False, comment="原始字节数"),
        sa.Column("media_type", ASCII_MEDIA_TYPE, nullable=True, comment="MIME 类型"),
        sa.Column("storage_uri", UTF8_STORAGE_URI, nullable=False, comment="持久内容 URI"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint("byte_size >= 0", name="ck_content_blobs_byte_size"),
        sa.PrimaryKeyConstraint("blob_id"),
        **MYSQL_TABLE_OPTIONS,
    )

    op.create_table(
        "documents",
        sa.Column("document_id", ASCII_UUID, nullable=False, comment="持久逻辑文档 UUID"),
        sa.Column("user_id", ASCII_USER_ID, nullable=False, comment="不透明认证主体 ID"),
        sa.Column("source_type", ASCII_SOURCE_TYPE, nullable=False, comment="来源类型"),
        sa.Column("display_name", UTF8_DISPLAY_NAME, nullable=False, comment="用户可见名称"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("document_id"),
        sa.UniqueConstraint(
            "user_id",
            "source_type",
            "display_name",
            name="uq_documents_owner_source_name",
        ),
        **MYSQL_TABLE_OPTIONS,
    )
    op.create_index("ix_documents_user_id", "documents", ["user_id"], unique=False)

    op.create_table(
        "document_revisions",
        sa.Column("document_id", ASCII_UUID, nullable=False),
        sa.Column("revision", UNSIGNED_INTEGER, nullable=False, comment="从 1 开始的单调版本"),
        sa.Column("blob_id", ASCII_SHA256, nullable=True),
        sa.Column(
            "normalized_text_sha256",
            ASCII_SHA256,
            nullable=True,
            comment="规范化全文 SHA-256",
        ),
        sa.Column(
            "normalized_text_uri",
            UTF8_STORAGE_URI,
            nullable=True,
            comment="规范化 UTF-8 文本的持久 URI",
        ),
        sa.Column(
            "legacy_index_only",
            sa.Boolean(),
            server_default=sa.text("0"),
            nullable=False,
            comment="历史索引缺少原件或区间",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint("revision >= 1", name="ck_document_revisions_revision"),
        sa.CheckConstraint(
            "legacy_index_only = 1 OR "
            "(blob_id IS NOT NULL AND normalized_text_sha256 IS NOT NULL "
            "AND normalized_text_uri IS NOT NULL)",
            name="ck_document_revisions_provenance",
        ),
        sa.ForeignKeyConstraint(
            ["blob_id"],
            ["content_blobs.blob_id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.document_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("document_id", "revision"),
        sa.UniqueConstraint(
            "document_id",
            "revision",
            "legacy_index_only",
            name="uq_document_revisions_identity_legacy",
        ),
        **MYSQL_TABLE_OPTIONS,
    )

    op.create_table(
        "index_versions",
        sa.Column("index_version", ASCII_SHA256, nullable=False, comment="规范配置 SHA-256"),
        sa.Column(
            "contract_version",
            sa.Integer(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.Column("config_json", sa.JSON(), nullable=False, comment="已规范化的索引配置"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "contract_version = 1",
            name="ck_index_versions_contract_version",
        ),
        sa.PrimaryKeyConstraint("index_version"),
        **MYSQL_TABLE_OPTIONS,
    )

    op.create_table(
        "index_chunks",
        sa.Column("chunk_id", ASCII_SHA256, nullable=False, comment="确定性 chunk SHA-256"),
        sa.Column("document_id", ASCII_UUID, nullable=False),
        sa.Column("document_revision", UNSIGNED_INTEGER, nullable=False),
        sa.Column("index_version", ASCII_SHA256, nullable=False),
        sa.Column("chunk_ordinal", UNSIGNED_INTEGER, nullable=False, comment="从 0 开始的切片序号"),
        sa.Column("content_sha256", ASCII_SHA256, nullable=False, comment="切片文本 UTF-8 SHA-256"),
        sa.Column("page_number", UNSIGNED_INTEGER, nullable=True, comment="1-based 页码"),
        sa.Column("char_start", UNSIGNED_INTEGER, nullable=True, comment="0-based 右开区间起点"),
        sa.Column("char_end", UNSIGNED_INTEGER, nullable=True, comment="0-based 右开区间终点"),
        sa.Column(
            "legacy_index_only",
            sa.Boolean(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint("document_revision >= 1", name="ck_index_chunks_revision"),
        sa.CheckConstraint("chunk_ordinal >= 0", name="ck_index_chunks_ordinal"),
        sa.CheckConstraint(
            "page_number IS NULL OR page_number >= 1",
            name="ck_index_chunks_page",
        ),
        sa.CheckConstraint(
            "(char_start IS NULL AND char_end IS NULL) OR "
            "(char_start IS NOT NULL AND char_end IS NOT NULL "
            "AND char_start >= 0 AND char_end > char_start)",
            name="ck_index_chunks_character_span",
        ),
        sa.CheckConstraint(
            "legacy_index_only = 1 OR "
            "(char_start IS NOT NULL AND char_end IS NOT NULL)",
            name="ck_index_chunks_nonlegacy_span",
        ),
        sa.ForeignKeyConstraint(
            ["document_id", "document_revision", "legacy_index_only"],
            [
                "document_revisions.document_id",
                "document_revisions.revision",
                "document_revisions.legacy_index_only",
            ],
            name="fk_index_chunks_document_revision",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["index_version"],
            ["index_versions.index_version"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("chunk_id"),
        sa.UniqueConstraint(
            "document_id",
            "document_revision",
            "index_version",
            "chunk_ordinal",
            name="uq_index_chunks_position",
        ),
        **MYSQL_TABLE_OPTIONS,
    )
    op.create_index(
        "ix_index_chunks_document_revision",
        "index_chunks",
        ["document_id", "document_revision"],
        unique=False,
    )
    op.create_index(
        "ix_index_chunks_index_version",
        "index_chunks",
        ["index_version"],
        unique=False,
    )


def downgrade() -> None:
    """仅在隔离库中按依赖逆序删除 M2 事实表。"""
    op.drop_index("ix_index_chunks_index_version", table_name="index_chunks")
    op.drop_index("ix_index_chunks_document_revision", table_name="index_chunks")
    op.drop_table("index_chunks")
    op.drop_table("index_versions")
    op.drop_table("document_revisions")
    op.drop_index("ix_documents_user_id", table_name="documents")
    op.drop_table("documents")
    op.drop_table("content_blobs")
