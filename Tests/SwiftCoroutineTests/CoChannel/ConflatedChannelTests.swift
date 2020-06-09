//
//  ConflatedChannelTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 08.06.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class ConflatedChannelTests: XCTestCase {
    
    class Test {}
    
    func testMeasure() {
        let group = DispatchGroup()
        measure {
            var counter = 0
            let channel = _ConflatedChannel<Int>()
            group.enter()
            DispatchQueue.global().async {
                DispatchQueue.concurrentPerform(iterations: 100_000) { index in
                    try? channel.awaitSend(index)
                }
                _ = channel.close()
            }
            ImmediateScheduler().startCoroutine {
                while (try? channel.awaitReceive()) != nil {
                    atomicAdd(&counter, value: 1)
                }
                group.leave()
            }
            group.wait()
            XCTAssertGreaterThanOrEqual(counter, 0)
        }
    }
    
    func testChannel() {
        let channel = _ConflatedChannel<Int>()
        XCTAssertEqual(channel.bufferType, .conflated)
        try? channel.awaitSend(1)
        try? channel.awaitSend(2)
        try? channel.awaitSend(3)
        XCTAssertEqual(channel.count, 1)
        XCTAssertFalse(channel.isEmpty)
        XCTAssertEqual(channel.poll(), 3)
        XCTAssertEqual(channel.count, 0)
        XCTAssertTrue(channel.isEmpty)
        XCTAssertNil(channel.poll())
        channel.sendFuture(.init(result: .success(4)))
        XCTAssertEqual(channel.poll(), 4)
    }
    
    func testReceive() {
        let exp = expectation(description: "testReceive")
        exp.expectedFulfillmentCount = 2
        let channel = _ConflatedChannel<Int>()
        channel.whenReceive {
            XCTAssertEqual($0, .success(1))
            exp.fulfill()
        }
        try? channel.awaitSend(1)
        try? channel.awaitSend(2)
        channel.whenReceive {
            XCTAssertEqual($0, .success(2))
            exp.fulfill()
        }
        wait(for: [exp], timeout: 1)
    }
    
    func testAwait() {
        let exp = expectation(description: "testAwait")
        let channel = _ConflatedChannel<Int>()
        ImmediateScheduler().startCoroutine {
            XCTAssertEqual(try? channel.awaitReceive(), 1)
            exp.fulfill()
        }
        try? channel.awaitSend(1)
        wait(for: [exp], timeout: 1)
    }
    
    func testInsideCoroutine() {
        let exp = expectation(description: "testInsideCoroutine")
        let channel = _ConflatedChannel<Int>()
        DispatchQueue.global().startCoroutine {
            var value = 0
            while let received = try? channel.awaitReceive() {
                value = received
            }
            XCTAssertEqual(value, 100_000)
            exp.fulfill()
        }
        for index in (0...100_000) {
            try? channel.awaitSend(index)
        }
        XCTAssertTrue(channel.close())
        wait(for: [exp], timeout: 20)
    }
    
    func testClose() {
        let channel = _ConflatedChannel<Int>()
        XCTAssertTrue(channel.offer(1))
        try? channel.awaitSend(1)
        XCTAssertFalse(channel.isClosed)
        XCTAssertTrue(channel.close())
        XCTAssertTrue(channel.isClosed)
        XCTAssertFalse(channel.close())
        XCTAssertFalse(channel.offer(2))
        XCTAssertEqual(channel.poll(), 1)
        XCTAssertNil(channel.poll())
        XCTAssertThrowError(CoChannelError.closed, channel.awaitReceive)
        XCTAssertThrowError(CoChannelError.closed) { try channel.awaitSend(1) }
    }
    
    func testClose2() {
        let exp = expectation(description: "testClose2")
        exp.expectedFulfillmentCount = 2
        let channel = _ConflatedChannel<Int>()
        ImmediateScheduler().startCoroutine {
            XCTAssertThrowError(CoChannelError.closed, channel.awaitReceive)
            exp.fulfill()
        }
        XCTAssertTrue(channel.close())
        channel.whenReceive {
            XCTAssertEqual($0, .failure(CoChannelError.closed))
            exp.fulfill()
        }
        wait(for: [exp], timeout: 1)
    }
    
    func testClose3() {
        let exp = expectation(description: "testClose3")
        let channel = _ConflatedChannel<Int>()
        channel.whenFinished { _ in exp.fulfill() }
        XCTAssertTrue(channel.close())
        wait(for: [exp], timeout: 1)
    }
    
    func testCancel() {
        let channel = _ConflatedChannel<Int>()
        XCTAssertTrue(channel.offer(1))
        XCTAssertFalse(channel.isCanceled)
        channel.cancel()
        channel.cancel()
        XCTAssertTrue(channel.isCanceled)
        XCTAssertNil(channel.poll())
        XCTAssertThrowError(CoChannelError.canceled, channel.awaitReceive)
        XCTAssertThrowError(CoChannelError.canceled) { try channel.awaitSend(1) }
    }
    
    func testCancel2() {
        let exp = expectation(description: "testCancel2")
        exp.expectedFulfillmentCount = 2
        let channel = _ConflatedChannel<Int>()
        ImmediateScheduler().startCoroutine {
            XCTAssertThrowError(CoChannelError.canceled, channel.awaitReceive)
            exp.fulfill()
        }
        channel.cancel()
        channel.whenReceive {
            XCTAssertEqual($0, .failure(CoChannelError.canceled))
            exp.fulfill()
        }
        wait(for: [exp], timeout: 1)
    }
    
    func testDeinit() {
        let exp = expectation(description: "testDeinit")
        var test: Test! = .init()
        var channel: _ConflatedChannel<Test>! = .init()
        weak var _weak = channel
        _ = channel.offer(test)
        channel = nil
        test = nil
        DispatchQueue.global().asyncAfter(deadline: .now() + 1) {
            XCTAssertNil(_weak)
            XCTAssertNil(test)
            exp.fulfill()
        }
        wait(for: [exp], timeout: 3)
    }
    
    func testDeinit2() {
        let exp = expectation(description: "testDeinit2")
        exp.expectedFulfillmentCount = 2
        var channel: _ConflatedChannel<Test>! = .init()
        channel.whenReceive { _ in exp.fulfill() }
        channel.whenReceive { _ in exp.fulfill() }
        channel = nil
        wait(for: [exp], timeout: 3)
    }
    
}
