#pragma once

#include <cassert>
#include <chrono>
#include <future>
#include <limits>
#include <utility>

#include "buffer_manager.h"
#include "buffer_transfer_manager.h"
#include "command.h"
#include "host_queue.h"
#include "logger.h"

namespace celerity {
namespace detail {

	class device_queue;
	class executor;
	class task_manager;

	class worker_job;

	class worker_job {
	  public:
		worker_job(command_pkg pkg, std::shared_ptr<logger> job_logger) : pkg(pkg), job_logger(job_logger) {}
		worker_job(const worker_job&) = delete;
		worker_job(worker_job&&) = delete;

		virtual ~worker_job() = default;

		void start();
		void update();

		bool is_running() const { return running; }
		bool is_done() const { return done; }

	  private:
		command_pkg pkg;
		std::shared_ptr<logger> job_logger;
		bool running = false;
		bool done = false;

		// Benchmarking
		std::chrono::steady_clock::time_point start_time;
		std::chrono::microseconds bench_sum_execution_time = {};
		size_t bench_sample_count = 0;
		std::chrono::microseconds bench_min = std::numeric_limits<std::chrono::microseconds>::max();
		std::chrono::microseconds bench_max = std::numeric_limits<std::chrono::microseconds>::min();

		virtual bool execute(const command_pkg& pkg, std::shared_ptr<logger> logger) = 0;

		/**
		 * Returns the job description in the form of the command_type, as well as a string describing the parameters.
		 * Used for logging.
		 */
		virtual std::pair<command_type, std::string> get_description(const command_pkg& pkg) = 0;
	};

	class horizon_job : public worker_job {
	  public:
		horizon_job(command_pkg pkg, std::shared_ptr<logger> job_logger) : worker_job(pkg, job_logger) { assert(pkg.cmd == command_type::HORIZON); }

	  private:
		bool execute(const command_pkg& pkg, std::shared_ptr<logger> logger) override { return true; };
		std::pair<command_type, std::string> get_description(const command_pkg& pkg) override;
	};

	/**
	 * Informs the data_transfer_manager about the awaited push, then waits until the transfer has been received and completed.
	 */
	class await_push_job : public worker_job {
	  public:
		await_push_job(command_pkg pkg, std::shared_ptr<logger> job_logger, buffer_transfer_manager& btm) : worker_job(pkg, job_logger), btm(btm) {
			assert(pkg.cmd == command_type::AWAIT_PUSH);
		}

	  private:
		buffer_transfer_manager& btm;
		std::shared_ptr<const buffer_transfer_manager::transfer_handle> data_handle = nullptr;

		bool execute(const command_pkg& pkg, std::shared_ptr<logger> logger) override;
		std::pair<command_type, std::string> get_description(const command_pkg& pkg) override;
	};

	class push_job : public worker_job {
	  public:
		push_job(command_pkg pkg, std::shared_ptr<logger> job_logger, buffer_transfer_manager& btm, buffer_manager& bm)
		    : worker_job(pkg, job_logger), btm(btm), buffer_mngr(bm) {
			assert(pkg.cmd == command_type::PUSH);
		}

	  private:
		buffer_transfer_manager& btm;
		buffer_manager& buffer_mngr;
		std::shared_ptr<const buffer_transfer_manager::transfer_handle> data_handle = nullptr;

		bool execute(const command_pkg& pkg, std::shared_ptr<logger> logger) override;
		std::pair<command_type, std::string> get_description(const command_pkg& pkg) override;
	};

	// host-compute jobs, master-node tasks and collective host tasks
	class host_execute_job : public worker_job {
	  public:
		host_execute_job(command_pkg pkg, std::shared_ptr<logger> job_logger, detail::host_queue& queue, detail::task_manager& tm, buffer_manager& bm)
		    : worker_job(pkg, job_logger), queue(queue), task_mngr(tm), buffer_mngr(bm) {
			assert(pkg.cmd == command_type::TASK);
		}

	  private:
		detail::host_queue& queue;
		detail::task_manager& task_mngr;
		detail::buffer_manager& buffer_mngr;
		std::future<detail::host_queue::execution_info> future;
		bool submitted = false;

		bool execute(const command_pkg& pkg, std::shared_ptr<logger> logger) override;
		std::pair<command_type, std::string> get_description(const command_pkg& pkg) override;
	};

	/**
	 * TODO: Optimization opportunity: If we don't have any outstanding await-pushes, submitting the kernel to SYCL right away may be faster,
	 * as it can already start copying buffers to the device (i.e. let SYCL do the scheduling).
	 */
	class device_execute_job : public worker_job {
	  public:
		device_execute_job(command_pkg pkg, std::shared_ptr<logger> job_logger, detail::device_queue& queue, detail::task_manager& tm, buffer_manager& bm)
		    : worker_job(pkg, job_logger), queue(queue), task_mngr(tm), buffer_mngr(bm) {
			assert(pkg.cmd == command_type::TASK);
		}

	  private:
		detail::device_queue& queue;
		detail::task_manager& task_mngr;
		detail::buffer_manager& buffer_mngr;
		cl::sycl::event event;
		bool submitted = false;

		std::future<void> computecpp_workaround_future;

		bool execute(const command_pkg& pkg, std::shared_ptr<logger> logger) override;
		std::pair<command_type, std::string> get_description(const command_pkg& pkg) override;
	};

} // namespace detail
} // namespace celerity
