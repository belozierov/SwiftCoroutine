//
//  CoroutineTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 21.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoroutineTests: XCTestCase {

    struct ImmediateScheduler: CoroutineScheduler {
        func scheduleTask(_ task: @escaping () -> Void) { task() }
    }
    
    func testNested() {
        let exp = expectation(description: "testAwait")
        exp.expectedFulfillmentCount = 2
        let dispatcher = CoroutineDispatcher.default
        DispatchQueue.global().async {
            dispatcher.execute(on: ImmediateScheduler()) {
               let coroutine = Coroutine.current
               dispatcher.execute(on: ImmediateScheduler()) {
                    XCTAssertFalse(coroutine === Coroutine.current)
                    Coroutine.await {
                        DispatchQueue.global().asyncAfter(deadline: .now() + 1, execute: $0)
                    }
                    XCTAssertFalse(coroutine === Coroutine.current)
                    exp.fulfill()
               }
               XCTAssertTrue(coroutine === Coroutine.current)
               exp.fulfill()
            }
        }
        wait(for: [exp], timeout: 3)
    }
    
    func testEscaping() {
        let exp = expectation(description: "testAwait")
        exp.expectedFulfillmentCount = 1000
        for i in 0..<1000 {
            DispatchQueue.main.startCoroutine {
                var a = Int.random(in: 0..<1000)
                a += Coroutine.await { completion in
                    if i == 0 { sleep(1) }
                    a += Int.random(in: 0..<1000)
                    DispatchQueue.global().asyncAfter(deadline: .now() + 1) {
                        a += Int.random(in: 0..<1000)
                        completion(Int.random(in: 0..<1000))
                    }
                }
                exp.fulfill()
            }
        }
        wait(for: [exp], timeout: 10)
    }
    
    func testStackSize() {
        XCTAssertEqual(Coroutine.StackSize(size: 234).size, 234)
        XCTAssertEqual(Coroutine.StackSize.pages(4).size, 4 * .pageSize)
    }
    
    func testDelay() {
        let exp = expectation(description: "testDelay")
        let date = Date()
        DispatchQueue.global().startCoroutine {
            Coroutine.delay(.seconds(1))
            XCTAssertDuration(from: date, in: 1..<2)
            exp.fulfill()
        }
        wait(for: [exp], timeout: 3)
    }
    
    func testAwait() {
        let exp = expectation(description: "testAwait")
        let queue = DispatchQueue.global()
        var sum = 0
        queue.startCoroutine {
            Coroutine.await {
                sum += 1
                $0()
            }
            sum += Coroutine.await { completion in
                queue.async {
                    completion(1)
                    completion(2)
                }
            }
            sum += Coroutine.await { completion in
                queue.async {
                    completion(1, 0)
                }
            }.0
            sum += Coroutine.await { completion in
                queue.async {
                    completion(1, 0, 0)
                }
            }.0
            XCTAssertEqual(sum, 4)
            exp.fulfill()
        }
        wait(for: [exp], timeout: 1)
    }
    
    func testInsideCoroutine() {
        let exp = expectation(description: "testInsideCoroutine")
        XCTAssertFalse(Coroutine.isInsideCoroutine)
        DispatchQueue.global().startCoroutine {
            XCTAssertTrue(Coroutine.isInsideCoroutine)
            let current = Coroutine.current
            DispatchQueue.global().await {
                XCTAssertTrue(Coroutine.isInsideCoroutine)
                XCTAssertTrue(current === Coroutine.current)
            }
            XCTAssertTrue(Coroutine.isInsideCoroutine)
            XCTAssertTrue(current === Coroutine.current)
            exp.fulfill()
        }
        XCTAssertFalse(Coroutine.isInsideCoroutine)
        wait(for: [exp], timeout: 5)
    }

}
