//
//  CoroutineSchedulerTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 30.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoroutineSchedulerTests: XCTestCase {
    
    func testAwait() {
        let date = Date()
        let exp = expectation(description: "testAwait")
        DispatchQueue.global().startCoroutine {
            try DispatchQueue.global().await {
                try Coroutine.delay(.seconds(1))
            }
            XCTAssertDuration(from: date, in: 1..<2)
            exp.fulfill()
        }
        wait(for: [exp], timeout: 3)
    }
    
    func testCoAwait() {
        let date = Date()
        let exp = expectation(description: "testAwait")
        DispatchQueue.global().startCoroutine {
            try DispatchQueue.global().coroutineFuture {
                try Coroutine.delay(.seconds(1))
            }.await()
            XCTAssertDuration(from: date, in: 1..<2)
            exp.fulfill()
        }
        wait(for: [exp], timeout: 3)
    }
    
    func testActor() {
        let exp = expectation(description: "testActor")
        let actor = DispatchQueue.global().actor(of: Int.self) {
            var count = 0
            for i in $0.makeIterator() {
                count += i
            }
            XCTAssertEqual(count, 10_000)
            exp.fulfill()
        }
        DispatchQueue.concurrentPerform(iterations: 10_000) { _ in
            actor.offer(1)
        }
        actor.close()
        wait(for: [exp], timeout: 5)
    }
    
}
