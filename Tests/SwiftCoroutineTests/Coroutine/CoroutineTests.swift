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
    
    func testEscaping() {
        let exp = expectation(description: "testAwait")
        exp.expectedFulfillmentCount = 1000
        for i in 0..<1000 {
            DispatchQueue.main.startCoroutine {
                var a = Int.random(in: 0..<1000)
                a += try! Coroutine.await { completion in
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
            try? Coroutine.delay(.seconds(1))
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
            try Coroutine.await {
                sum += 1
                $0()
            }
            sum += try Coroutine.await { completion in
                queue.async {
                    completion(1)
                }
            }
            sum += try Coroutine.await { completion in
                queue.async {
                    completion(1, 0)
                }
            }.0
            sum += try Coroutine.await { completion in
                queue.async {
                    completion(1, 0, 0)
                }
            }.0
            XCTAssertEqual(sum, 4)
            exp.fulfill()
        }
        wait(for: [exp], timeout: 1)
    }

}
