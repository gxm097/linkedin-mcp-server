"""
LinkedIn post/content search and comment tools.

search_posts performs LinkedIn's global content search (the "Posts" results
tab) using innerText extraction, so informal "we're hiring" / "Buscamos ..."
posts can be found before a formal job listing is published. Mirrors
search_people: build a /search/results/content/ URL, scroll to load results,
and return the raw innerText for the LLM to parse, plus post-permalink
references.

comment_on_post is the write side: it navigates to one post's permalink and
types into that post's own comment box. It is gated on confirm_comment the way
send_message is gated on confirm_send.
"""

import logging
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.dependencies import get_ready_extractor, tool_errors
from linkedin_mcp_server.linkedin.contracts import FilterValidationError

logger = logging.getLogger(__name__)


def register_post_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register post/content-search tools with the MCP server."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Search Posts",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"post", "search"},
    )
    @tool_errors
    async def search_posts(
        keywords: str,
        ctx: Context,
        date_posted: str | None = None,
        max_pages: Annotated[int, Field(ge=1, le=10)] = 3,
    ) -> dict[str, Any]:
        """
        Search LinkedIn posts/content globally by keyword (the "Posts" tab).

        Use this to catch informal hiring posts ("we're hiring", "Buscamos
        ...", "estamos contratando", "join our team") that often appear before
        a formal job listing exists. This is global content search, distinct
        from get_feed (your own home feed) and get_company_posts (one
        company's page).

        Args:
            keywords: Search keywords (e.g., "Buscamos Unity", "AI automation hiring")
            ctx: FastMCP context for progress reporting
            date_posted: Optional recency filter. One of "past-24h",
                "past-week", "past-month"; the "past_24_hours" / "past_week" /
                "past_month" spellings used by search_jobs are accepted too.
                Omit for any time.
            max_pages: Scroll depth as result "pages" of ~5 scrolls each
                (1-10, default 3). Content search is an infinite scroll, so
                this caps how far the page is scrolled rather than fetching
                discrete pages.

        Returns:
            Dict with url, sections (search_results -> raw text), and optional
            references (post authors, companies, linked jobs, and kind
            "feed_post" permalinks read from the page's payload responses —
            /feed/update/<urn>/ or /posts/<slug>, both valid permalinks) and
            section_errors. The DOM carries no per-post permalink anchors;
            captured permalinks are not aligned to result order. The LLM
            should parse the raw text to extract each post's author,
            headline/role, company, body, posted date, and reaction/comment
            counts.
        """
        extractor = await get_ready_extractor(ctx, tool_name="search_posts")
        logger.info(
            "Searching posts: keywords='%s', date_posted='%s', max_pages=%d",
            keywords,
            date_posted,
            max_pages,
        )

        await ctx.report_progress(progress=0, total=100, message="Starting post search")

        try:
            result = await extractor.search_posts(
                keywords,
                date_posted=date_posted,
                max_pages=max_pages,
            )
        except FilterValidationError as e:
            # Validation messages carry actionable detail; surface them as
            # ToolError so mask_error_details doesn't reduce them to a
            # generic "Error calling tool 'search_posts'".
            raise ToolError(str(e)) from e

        await ctx.report_progress(progress=100, total=100, message="Complete")

        return result

    @mcp.tool(
        timeout=tool_timeout,
        title="Comment On Post",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"post", "actions"},
    )
    @tool_errors
    async def comment_on_post(
        post: str,
        comment: str,
        confirm_comment: bool,
        ctx: Context,
    ) -> dict[str, Any]:
        """
        Comment on a LinkedIn post as the signed-in account.

        The post is identified by its permalink, and the permalink alone decides
        where the comment lands: the tool navigates there and refuses to type
        unless the browser's own address is still that post. Pass a reference
        exactly as get_feed, search_posts or get_company_posts returned it in
        ``references``. Note that search_posts appends permalinks without
        aligning them to result order, so a permalink taken from a search must
        be matched to the post you meant some other way before commenting: dry
        run it first and read the ``excerpt`` back, which is the post's own
        text from the page the comment would land on.

        Only the post's own comment box is used, and only while it is the single
        comment box open on the page and already empty. Text someone else typed
        is never overwritten, and a reply to an existing comment is not
        reachable from here.

        With confirm_comment False this is a read: it reports whether the post
        accepts a comment from this account, hands back an ``excerpt`` of that
        post's text, and returns ``dry_run`` without typing anything. This is a
        write operation when confirm_comment is True.

        Commenting is rate-limited by LinkedIn and automated commenting at
        volume is what account restrictions are for. Treat each call as a
        deliberate act the user asked for, not as a loop to run over a list.

        Args:
            post: The post's permalink or entity URN. Accepts
                ``urn:li:activity:…``, ``/feed/update/<urn>/``,
                ``/posts/<slug>`` and the full URL form of either.
            comment: Comment text. LF, CRLF and CR all end a line; an empty line
                separates paragraphs and is kept. Whitespace-only lines count as
                empty, and leading and trailing whitespace of the whole comment
                is removed, as LinkedIn does when posting. Tab, other control
                characters, DEL, U+0085, U+2028 and U+2029 are refused.
            confirm_comment: Must be True to post the comment
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with url, status, message, posted and retry_safe, plus
            ``excerpt`` and ``excerpt_truncated`` on a ``dry_run``. The excerpt
            is the post's own text, read off the permalink under the same route
            guard the comment is held to, and it is how a caller confirms it
            reached the post it meant; it is null when the text could not be
            read, and ``excerpt_truncated`` true means the post runs on past
            it, so a phrase missing from the excerpt is not missing from the
            post. ``posted`` is
            true only once the comment was observed on the post; it is false
            both where nothing was submitted and where the outcome is unknown.
            ``retry_safe`` separates the two: it is false from the moment the
            post button is clicked, and calling again while it is false can
            leave the comment on the post twice. A ``status`` of
            ``outcome_unknown`` means the click happened but the comment was not
            seen — open the post and check rather than retrying.
        """
        extractor = await get_ready_extractor(ctx, tool_name="comment_on_post")
        logger.info(
            "Commenting on post: %s (confirmed=%s)",
            post,
            confirm_comment,
        )

        await ctx.report_progress(progress=0, total=100, message="Opening the post")

        result = await extractor.comment_on_post(
            post,
            comment,
            confirm_comment=confirm_comment,
        )

        await ctx.report_progress(progress=100, total=100, message="Complete")

        return result
