//
//  NIOAsyncTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 21.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import NIO
import XCTest
import NIOConcurrencyHelpers
import SwiftCoroutine

final class NIOAsyncTests: XCTestCase {
    var elg: EventLoopGroup!

    override func setUp() {
        super.setUp()

        self.elg = MultiThreadedEventLoopGroup(numberOfThreads: 4)
    }

    override func tearDown() {
        do {
            try self.elg.syncShutdownGracefully()
        } catch let error {
            XCTFail("- [ELG SHUTDOWN FAILURE]: \(error)")
        }

        super.tearDown()
    }

    func testAsyncQuick() throws {
        let eventLoop = self.elg.next()
        let list = NumberList()

        let result = eventLoop.async {
            for number in (1...9).map({ eventLoop.makeSucceededFuture($0) }) {
                try list.append(number.await())
            }
        }

        try XCTAssertNoThrow(result.wait())
        XCTAssertEqual(list.numbers, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    }

    func testMultipleAsync() throws {
        let eventLoop = self.elg.next()
        let relatedFutures = (0...9).map { _ in
            return eventLoop.async { () throws -> () in
                for future in (0...9).map({ _ in eventLoop.makeSucceededFuture(()) }) {
                    try future.await()
                }
            }
        }

        try XCTAssertNoThrow(relatedFutures.map { try $0.wait() })
        measure {
            for _ in 0..<100 {
                let unrelatedFutures = (0...9).map { t -> EventLoopFuture<Void> in
                    let next = self.elg.next()
                    return next.async { () throws -> () in
                        for future in (0...9).map({ _ in eventLoop.makeSucceededFuture(()) }) {
                            try future.await()
                        }
                    }
                }

                try? XCTAssertNoThrow(unrelatedFutures.map { try $0.wait() })
            }
        }
    }

    func testAsyncOverheadMetric() throws {
        let eventLoop = self.elg.next()
        let range = (0..<10_000).map { eventLoop.makeSucceededFuture($0) }

        measure {
            let result = eventLoop.async {
                for number in range {
                    _ = try self.instantReturn(number.await())
                }
            }

            do {
                try result.wait()
            } catch let error {
                XCTFail(error.localizedDescription)
            }
        }
    }

    func testAsyncMultithreadedOverheadMetric() throws {
        let eventLoops = (0...3).map { _ in self.elg.next() }

        let ranges = [
            (0..<2_500).map { eventLoops[0].makeSucceededFuture($0) },
            (2_500..<5_000).map { eventLoops[1].makeSucceededFuture($0) },
            (5_000..<7_500).map { eventLoops[2].makeSucceededFuture($0) },
            (7_500..<10_000).map { eventLoops[3].makeSucceededFuture($0) }
        ]

        measure {
            let results = (0...3).map { index in
                return eventLoops[index].async {
                    for number in ranges[index] {
                        _ = try self.instantReturn(number.await())
                    }
                }
            }

            do {
                let promise = self.elg.next().makePromise(of: Void.self)
                let count = NIOAtomic.makeAtomic(value: 1)

                results.forEach { future in
                    future.whenComplete { result in
                        switch result {
                        case .success:
                            _ = count.add(1)
                            if count.load() == 4 { promise.succeed(()) }
                        case let .failure(error):
                            promise.fail(error)
                        }
                    }
                }

                try promise.futureResult.wait()
            } catch let error {
                XCTFail(error.localizedDescription)
            }
        }
    }

    func instantReturn(_ number: Int) -> Int { return number }
}

final class NumberList {
    private let lock: Lock
    private(set) var numbers: [Int] {
        didSet { print(self.numbers) }
    }

    init() {
        self.lock = Lock()
        self.numbers = []
    }

    func append(_ number: Int) {
        self.lock.withLockVoid { self.numbers.append(number) }
    }
}

extension EventLoop {
    func future<Value>(_ value: Value, after: UInt32) -> EventLoopFuture<Value> {
        sleep(after)
        return self.makeSucceededFuture(value)
    }
}

