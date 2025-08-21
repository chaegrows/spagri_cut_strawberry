from typing import Optional, Dict, Any, Type
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import copy

from config.common.statistics_collector.commander_status import CommanderStatusFrozen

# --- Pollination Statistics ---


class BaseStatisticsWorking:
    def __init__(self):
        self.n_success: int = 0
        self.n_skipped: int = 0
        self.start_time: datetime = datetime.now(timezone.utc)
        self.end_time: datetime = datetime.now(timezone.utc)
        self.additional_stats: Dict[str, Any] = {}

    def increment_success(self):
        self.n_success += 1

    def increment_skipped(self):
        self.n_skipped += 1

    def mark_start(self):
        self.start_time = datetime.now(timezone.utc)

    def mark_end(self):
        self.end_time = datetime.now(timezone.utc)

    def reset(self):
        """Reset all statistics."""
        self.n_success = 0
        self.n_skipped = 0
        self.start_time = datetime.now(timezone.utc)
        self.end_time = None
        self.additional_stats.clear()

    def add_stat(self, key: str, value: Any):
        """Add additional statistics."""
        self.additional_stats[key] = value

    def get_stat(self, key: str) -> Any:
        """Get additional statistics."""
        return self.additional_stats.get(key)

class BaseStatisticsFrozen(BaseModel):
    n_success: int
    n_skipped: int
    success_rate: float
    start_time: datetime
    end_time: datetime
    total_work_time: float
    avg_time_per_attempt: Optional[float]
    efficiency: Optional[float]
    additional_stats: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True

class StatisticsManager:
    def __init__(self):
        self.working_stats: Dict[str, BaseStatisticsWorking] = {}

    def get_working_stats(self, work_type: str) -> BaseStatisticsWorking:
        """Get working statistics for a specific work type."""
        if work_type not in self.working_stats:
            self.working_stats[work_type] = BaseStatisticsWorking()
        return self.working_stats[work_type]

    def increment_success(self, work_type: str):
        """Increment success count for a specific work type."""
        stats = self.get_working_stats(work_type)
        stats.increment_success()

    def increment_skipped(self, work_type: str):
        """Increment skipped count for a specific work type."""
        stats = self.get_working_stats(work_type)
        stats.increment_skipped()

    def mark_start(self, work_type: str):
        """Mark start time for a specific work type."""
        stats = self.get_working_stats(work_type)
        stats.mark_start()

    def mark_end(self, work_type: str):
        """Mark end time for a specific work type."""
        stats = self.get_working_stats(work_type)
        stats.mark_end()

    def reset(self, work_type: str):
        """Reset statistics for a specific work type."""
        stats = self.get_working_stats(work_type)
        stats.reset()

    def add_stat(self, work_type: str, key: str, value: Any):
        """Add additional statistics for a specific work type."""
        stats = self.get_working_stats(work_type)
        stats.add_stat(key, value)

    def get_stat(self, work_type: str, key: str) -> Any:
        """Get additional statistics for a specific work type."""
        stats = self.get_working_stats(work_type)
        return stats.get_stat(key)

    def export_frozen(self, work_type: str) -> BaseStatisticsFrozen:
        """Export frozen statistics for a specific work type."""
        if work_type not in self.working_stats:
            raise ValueError(f"No statistics available for work type: {work_type}")

        stats = self.working_stats[work_type]
        if stats.end_time is None:
            raise ValueError("End time not set.")

        total_attempts = stats.n_success + stats.n_skipped
        success_rate = (stats.n_success / total_attempts) * 100 if total_attempts > 0 else 0.0
        total_work_time = (datetime.now(timezone.utc) - stats.start_time).total_seconds()
        if total_work_time < 0: total_work_time = 0.0
        avg_time_per_attempt = total_work_time / total_attempts if total_attempts > 0 else -1
        if avg_time_per_attempt < 0: avg_time_per_attempt = 0.0
        efficiency = stats.n_success / (total_work_time / 60) if total_work_time > 0 else -1

        return BaseStatisticsFrozen(
            n_success=stats.n_success,
            n_skipped=stats.n_skipped,
            success_rate=success_rate,
            start_time=stats.start_time,
            end_time=stats.end_time,
            total_work_time=total_work_time,
            avg_time_per_attempt=avg_time_per_attempt,
            efficiency=efficiency,
            additional_stats=stats.additional_stats
        )


class StatisticsModel(BaseModel):
    commander_status: CommanderStatusFrozen
    statistics: BaseStatisticsFrozen

    class Config:
        frozen = True

def create_stats_model(
        commander_status: CommanderStatusFrozen,
        statistics: BaseStatisticsFrozen
        ) -> StatisticsModel:
    ret = StatisticsModel(
        commander_status=commander_status,
        statistics=statistics
    )
    return ret


if __name__ == "__main__":
    # 통계 관리자 초기화
    stats_manager = StatisticsManager()

    # 수분 작업 통계 수집 예시
    stats_manager.mark_start("pollination")
    stats_manager.increment_success("pollination")
    stats_manager.increment_success("pollination")
    stats_manager.increment_success("pollination")
    stats_manager.increment_success("pollination")
    stats_manager.add_stat("pollination", "flower_types", {"strawberry": 5, "tomato": 3})
    stats_manager.add_stat("pollination", "pollination_method", "manual")
    stats_manager.increment_skipped("pollination")
    stats_manager.mark_end("pollination")

    # 검사 작업 통계 수집 예시
    stats_manager.mark_start("inspection")
    stats_manager.increment_success("inspection")
    stats_manager.add_stat("inspection", "defect_count", {"disease": 2, "pest": 1})
    stats_manager.add_stat("inspection", "inspection_type", "visual")
    stats_manager.increment_skipped("inspection")
    stats_manager.mark_end("inspection")

    # 통계 내보내기 및 출력
    try:
        pollination_stats = stats_manager.export_frozen("pollination")
        inspection_stats = stats_manager.export_frozen("inspection")

        print("\n=== 수분 작업 통계 ===")
        print(f"성공: {pollination_stats.n_success}")
        print(f"건너뜀: {pollination_stats.n_skipped}")
        print(f"성공률: {pollination_stats.success_rate:.1f}%")
        print(f"총 작업 시간: {pollination_stats.total_work_time:.1f}초")
        print(f"평균 시도 시간: {pollination_stats.avg_time_per_attempt:.1f}초")
        print(f"효율성: {pollination_stats.efficiency:.1f} 작업/분")
        print(f"꽃 종류: {pollination_stats.additional_stats['flower_types']}")
        print(f"수분 방법: {pollination_stats.additional_stats['pollination_method']}")

        print("\n=== 검사 작업 통계 ===")
        print(f"성공: {inspection_stats.n_success}")
        print(f"건너뜀: {inspection_stats.n_skipped}")
        print(f"성공률: {inspection_stats.success_rate:.1f}%")
        print(f"총 작업 시간: {inspection_stats.total_work_time:.1f}초")
        print(f"평균 시도 시간: {inspection_stats.avg_time_per_attempt:.1f}초")
        print(f"효율성: {inspection_stats.efficiency:.1f} 작업/분")
        print(f"결함 수: {inspection_stats.additional_stats['defect_count']}")
        print(f"검사 유형: {inspection_stats.additional_stats['inspection_type']}")

    except ValueError as e:
        print(f"오류 발생: {e}")
